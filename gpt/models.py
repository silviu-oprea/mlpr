import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, emb_dim, head_dim, block_size):
        super().__init__()

        # Here we assume the input and output dimensions are both head_dim
        self.key_layer = torch.nn.Linear(emb_dim, head_dim, bias=False)
        self.query_layer = torch.nn.Linear(emb_dim, head_dim, bias=False)
        self.value_layer = torch.nn.Linear(emb_dim, head_dim, bias=False)

        self.dropout_layer = nn.Dropout(p=0.1)

        # This is not a parameter
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, input_ids):
        B, T, C = input_ids.size()

        query = self.query_layer(input_ids) # (B, T, C)
        key = self.key_layer(input_ids)     # (B, T, C)

        # Compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = query @ key.transpose(2, 1) / C**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout_layer(wei)

        # Weighted aggregation of the values
        value = self.value_layer(input_ids) # (B, T, C)
        # (B, T, T) @ (B, T, C) -> (B, T, C)
        out = wei @ value
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, head_dim, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(emb_dim, head_dim, block_size) for _ in range(num_heads)
        ])
        # Project back into the residuual pathway
        self.proj_layer = nn.Linear(emb_dim, emb_dim)
        self.dropout_layer = nn.Dropout(p=0.1)

    def forward(self, input_ids):
        # a list of tensors, each of dim
        # (batch_size, doc_len <= block_size, emb_dim)
        # (denoted (B, T, C) above for brevity)
        out = [head(input_ids) for head in self.heads]
        # (batch_size, doc_len, emb_dim * num_heads)
        out = torch.cat(out, dim=-1)
        out = self.dropout_layer(self.proj_layer(out))
        return out


class FeedForward(nn.Module):
    """A linear layer followed by a non-linearity"""

    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(p=0.1)
        )

    def forward(self, input_ids):
        return self.net(input_ids)


class Block(nn.Module):
    def __init__(self, num_self_attn_heads, emb_dim, block_size):
        super().__init__()
        head_dim = emb_dim // num_self_attn_heads
        self.self_attn_heads = MultiHeadAttention(
            num_self_attn_heads, emb_dim, head_dim, block_size
        )
        self.ff_layer = FeedForward(emb_dim)
        self.ln1_layer = nn.LayerNorm(emb_dim)
        self.ln2_layer = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # (B, T, C)
        x = x + self.self_attn_heads(self.ln1_layer(x))
        x = x + self.ff_layer(self.ln1_layer(x))
        return x


class BigramLanguageModel(torch.nn.Module):
    def __init__(
        self,
        num_self_attn_heads,
        emb_dim,
        block_size,
        vocab_size
    ):
        super().__init__()
        self.block_size = block_size

        self.tok_embed_layer = torch.nn.Embedding(vocab_size, emb_dim)
        self.pos_embed_layer = torch.nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(
            Block(num_self_attn_heads, emb_dim, block_size),
            Block(num_self_attn_heads, emb_dim, block_size),
            Block(num_self_attn_heads, emb_dim, block_size),
            nn.LayerNorm(emb_dim)
        )
        self.lm_head = torch.nn.Linear(emb_dim, vocab_size)

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids (`torch.LongTensor` of size
                       `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            targets (`torch.LondTensor` of size `(batch_size, sequence_length)`):
                Indices of target tokens in the vocabulary.
        """
        # (batch_size, doc_len, embed_dim)
        token_embeds = self.tok_embed_layer(input_ids)

        batch_size, doc_len = input_ids.size()
        # (doc_len,)
        pos_ids = torch.arange(doc_len, device=input_ids.device)
        # (doc_len, embed_dim)
        pos_embeds = self.pos_embed_layer(pos_ids)

        # When broadcasting, PyTorch right-aligns.
        #   (batch_size, doc_len, embed_dim) +
        #               (doc_len, embed_dim)
        # = (batch_size, doc_len, embed_dim).
        x = token_embeds + pos_embeds

        x = self.blocks(x)

        # (batch_size, seq_length, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # embed_dim = vocab_size so time_embeds can be considered logits.
            # The task is: given the current word, predict the next word.

            batch_size, seq_len, embed_dim = logits.size()
            logits = logits.view(batch_size * seq_len, embed_dim)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input_ids, max_new_tokens):
        """
        Args:
            input_ids (`torch.Tensor` of dim (batch_size, doc_len)): Initially,
                it could be the start-of-sequence token.
        """
        for _ in range(max_new_tokens):
            # (batch_size, seq_len, embed_dim)
            # Note we are only keeping the last block_size tokens, otherwise
            #   we'll run out of positional embeddings.
            logits, _ = self(input_ids[:, -self.block_size:])

            # Keep the logits for the last token only.
            # (batch_size, vocab_size == embed_dim)
            logits = logits[:, -1, :]

            # (batch_size, vocab_size)
            probs = F.softmax(logits, dim=1)

            # Sample from distribution (one sample (an index) from each row)
            # (batch_size, 1)
            next_ids = torch.multinomial(probs, num_samples=1)

            # Append the sample to the running sequence
            # (batch_size, seq_len + 1)
            input_ids = torch.cat((input_ids, next_ids), dim=1)
        return input_ids
