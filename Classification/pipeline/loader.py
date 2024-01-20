import json
# import re
# import os
# import numpy as np
import torch
from torch.utils.data import DataLoader     # Dataset
from transformers import BertTokenizer
# from config import Config


class DataGenerator:
    def __init__(self, config, data_path):
        super(DataGenerator, self).__init__()
        self.config = config
        self.train_valid_data_path = data_path    # "../train_tag_news.json" / "../valid_tag_news.json"
        self.chars_path = config["chars_path"]              # "../chars.txt"

        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])

        self.vocab_dict = build_vocab_from_chars(self.chars_path)
        # self.vocab_dict: {'[UNK]': 1, '[SPACE]': 2, '!': 3, '"': 4, '#': 5,....}

        self.config["vocab_size"] = len(self.vocab_dict)
        # print(self.config["vocab_size"]) : 4624

        # self.vocab_dict = build_vocab_from_data(config["data_path"])
        # 上面两种方法分别是已有字表文件和没有字表文件的处理

        self.label2index = build_label2index(self.train_valid_data_path)
        self.index2label = build_index2label(self.label2index)
        # print(self.label2index, self.index2label)
        # {'军事': 0, '游戏': 1, '时尚': 2, '家居': 3, '体育': 4, '房产': 5, '旅游': 6,  '社会': 17}
        # {0: '军事', 1: '游戏', 2: '时尚', 3: '家居', 4: '体育', 5: '房产', 6: '旅游',  17: '社会'}
        self.load()

    def load(self):
        self.data = []
        with open(self.train_valid_data_path, encoding='utf8') as f:
            for line in f:
                line = json.loads(line)
                data_x = line["title"] + line["content"]
                data_y = line["tag"]
                # print("data_x, data_y:", data_x, data_y)
                if self.config["model_type"] == "bert":
                    data_x2id = self.tokenizer.encode(data_x, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    data_x2id = self.sentence2index(data_x)
                data_y2id = self.label2id(data_y)
                # print("data_y2id:", data_y2id)
                data_x2id = torch.LongTensor(data_x2id)
                data_y2id = torch.LongTensor([data_y2id])
                self.data.append([data_x2id, data_y2id])
                # print(data_x2id, "+++++", data_y2id)
            # print("len(self.data):", len(self.data), len(self.data[0][0]), len(self.data[0][1]))

    def label2id(self, data_y):
        return self.label2index[data_y]

    def sentence2index(self, data_x):
        data_x2id = []
        for x in data_x:
            for char in x:
                data_x2id.append(self.vocab_dict.get(char, self.vocab_dict["UNK"]))
        data_x2id = self.padding2max_length(data_x2id)
        return data_x2id

    def padding2max_length(self, data_x2id):
        max_length = self.config["max_length"]
        data_x2id = data_x2id[: max_length]
        data_x2id += [0] * (max_length - len(data_x2id))
        return data_x2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def build_vocab_from_chars(chars_path):
    """

    :param chars_path: "../chars.txt"
    :return: word_to_index, type: dict
    """
    vocab_dict = {}
    with open(chars_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            line = line.strip()
            vocab_dict[line] = index + 1
        vocab_dict["UNK"] = index + 2
        vocab_dict["padding"] = 0
    # print(vocab_dict)
    return vocab_dict


def build_vocab_from_data(corpus_path):
    vocab_dict = {}
    index = 0
    with open(corpus_path, encoding="utf8") as f:
        for line in f:
            for char in line:
                if char not in vocab_dict:
                    vocab_dict[char] = index
                    index += 1
    vocab_dict["padding"] = 0
    return vocab_dict


def build_label2index(train_data_path):
    class_dict = {}
    tag_index = 0
    with open(train_data_path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            tag = line["tag"]
            if tag not in class_dict:
                class_dict[tag] = tag_index
                tag_index += 1
    return class_dict


def build_index2label(label2index):
    index2label = dict((y, x) for x, y in label2index.items())
    return index2label


def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(config, data_path)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
