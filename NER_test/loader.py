import json
import torch
import jieba
from torch.utils.data import Dataset, DataLoader


class DataGenerator:
    def __init__(self, data_path, config):
        self.path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_len"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.sentences = []
        self.data = []
        self.load()

    def load(self):
        with open(self.path, encoding="utf8") as f:
            sentences = f.read().split("\n\n")
            for sentence in sentences[:-1]:
                sentence_list = []
                label_list = []
                lines = sentence.split("\n")
                for line in lines:
                    char, label = line.split()
                    sentence_list.append(char)
                    label_list.append(self.schema[label])

                self.sentences.append("".join(sentence_list))
                input_ids = self.sentence2id(sentence_list)
                input_ids = self.padding(input_ids)
                label = self.padding(label_list, padding_num=-1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(label)])
        # print("len(self.data)", len(self.data))
        return

    def sentence2id(self, sentence):
        input_ids = []
        for char in sentence:
            input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))
        return input_ids

    def padding(self, sentence, padding_num=0):
        res = sentence[: self.config["max_length"]]
        res += [padding_num] * (self.config["max_length"] - len(sentence))
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(path):
    vocab_dict = {}
    with open(path, encoding="utf8") as f:
        for index, char in enumerate(f):
            char = char.strip()
            vocab_dict[char] = index + 1
    return vocab_dict


def load_schema(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def data_loader(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    # print(len(dl))
    return dl
