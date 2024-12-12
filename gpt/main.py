import argparse
import tqdm
import torch

from models import BigramLanguageModel


torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data/tiny_shakespeare.txt")
args = parser.parse_args()

# ============================================================================ #
# Hyperparameters
emb_dim = 32
batch_size = 32  # Number of independent sequences processed in parallel
block_size = 8  # Maximum context length
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_train_epochs = 5000
val_freq = 500
gen_max_new_tokens = 100
num_self_attn_heads = 4

# ============================================================================ #
# Dataset

with open(args.data_path, "r", encoding="utf-8") as fp:
    corpus = fp.read()

vocab = sorted(list(set(corpus)))
c_to_i = {c: i for i, c in enumerate(vocab)}
i_to_c = {i: c for i, c in enumerate(vocab)}
encode = lambda txt: [c_to_i[c] for c in txt]
decode = lambda idx_lst: ''.join([i_to_c[i] for i in idx_lst])

txt = "hello world!"
print(encode(txt))
print(decode(encode(txt)))

data = torch.tensor(encode(corpus), dtype=torch.long)

# Split the data into a train and validation sets
num_train_ex = int(0.9 * len(data))
train_data = data[:num_train_ex]
val_data = data[num_train_ex:]

def get_batch(data):
    start_positions = torch.randint(len(data) - block_size, (batch_size,))
    input = torch.stack([data[i: i + block_size] for i in start_positions])
    output = torch.stack(
        [data[i + 1: i + 1 + block_size] for i in start_positions]
    )
    return input, output

get_train_batch = lambda: get_batch(train_data)
get_val_batch = lambda: get_batch(val_data)

# A train batch looks like:
# (tensor([[51, 59, 57, 58,  1, 39, 61, 39],
#         [63,  6,  1, 61, 46, 53,  5, 57],
#         [39,  1, 41, 46, 47, 50, 42, 10],
#         [53, 59, 10,  0, 37, 53, 59,  1]])
#  tensor([[59, 57, 58,  1, 39, 61, 39, 63],
#         [ 6,  1, 61, 46, 53,  5, 57,  1],
#         [ 1, 41, 46, 47, 50, 42, 10,  0],
#         [59, 10,  0, 37, 53, 59,  1, 57]]))

# ============================================================================ #
# Model

model = BigramLanguageModel(
    num_self_attn_heads, emb_dim, block_size, len(vocab)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
gen_input_ids = torch.zeros((1, 1), dtype=torch.long)

losses = []

for step in range(num_train_epochs):

    input_ids, targets = get_train_batch()
    logits, loss = model(input_ids, targets)
    # zero all gradients from prev step
    optimizer.zero_grad(set_to_none=True)
    # compute gradients for all parameters
    loss.backward()
    # use those gradients to update parameters
    optimizer.step()
    losses.append(loss.item())

    if step % val_freq == 0:
        avg_loss = sum(losses) / len(losses)
        losses = []
        gen_ids = model.generate(
            gen_input_ids, max_new_tokens=gen_max_new_tokens
        )
        print(f'step: {step} - loss: {avg_loss:.2f}')
        print('sampling:', decode(gen_ids[0].tolist()))
        print("-" * 50)

print('last loss', loss.item())
gen_ids = model.generate(gen_input_ids, max_new_tokens=100)
print('Sampling after training', decode(gen_ids[0].tolist()))
