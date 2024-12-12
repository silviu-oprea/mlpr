
class Dataset(object):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as fp:
            self.data = fp.read()


class CharTokenizer(object):
    def __init__():

