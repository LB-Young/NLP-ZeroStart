# coding:utf8
import os
import torch
import torch.nn as nn
from torch.nn import functional
import numpy as np
import random
import json
from transformers import BertModel
import re
import logging
from config import Config
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyModel(nn.Module):
    def __init__(self, word_dim, category):
        super(MyModel, self).__init__()
        self.bert = BertModel.from_pretrained(r"E:\python project\badou_new\bert-base-chinese", return_dict=False)
        self.classification = nn.Linear(word_dim, category)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        sequence_output, cls_token = self.bert(x)
        category_output = self.classification(cls_token)
        category_output = self.softmax(category_output)
        # print(category_output)
        if y is not None:
            return cal_loss(category_output, y)
        else:
            return category_output


def cal_loss(category_output, y):
    # print(category_output, y)
    batch_size = len(y)
    ce_loss = 0
    for cate_index in range(len(y)):
        ce_loss += -(torch.log(category_output[cate_index][y[cate_index]]))
    # print(type(ce_loss))
    return torch.mean(ce_loss)


def build_word_dict(path):
    word_dict = {}
    val = 0
    with open(path, "r", encoding="utf8") as f:
        for index, sample in enumerate(f):
            # print(type(sample))
            for word in sample:
                if word not in word_dict:
                    word_dict[word] = val
                    val += 1
        word_dict["unk"] = val
        word_dict["padding"] = val + 1
    return word_dict


def build_sample(train_corpus_path, word_dict):
    """
    x[],y[]是文本内容，dataset_x[],dataset_y[]是文本对应的索引内容
    :param train_corpus_path:
    :param word_dict:
    :return:
    """
    x = []
    y = []
    tag_dict = {}
    tag_index = 0
    with open(train_corpus_path, "r", encoding="utf8") as f:
        for index, news in enumerate(f):
            tag = news[9: 11]
            title_content = re.sub("({\"tag\": \".{2}\", \"title\": \")|(\"})", "", news)
            title_content = re.sub("(\", \"content\": \")", "。", title_content)
            # print(index, tag, title_content)
            x.append(title_content)
            y.append(tag)
            if tag not in tag_dict:
                tag_dict[tag] = tag_index
                tag_index += 1
        # print(y, x)
        dataset_x = word2id(x, word_dict)
        dataset_y = tag2id(y, tag_dict)
    return dataset_x, dataset_y


def tag2id(y, tag_dict):
    dataset_y = []
    for tag in y:
        dataset_y.append(tag_dict[tag])
    return torch.LongTensor(dataset_y)


def word2id(x, word_dict):
    dataset_x = []
    sample_x = []
    for title_content in x:
        for char in title_content:
            sample_x.append(word_dict.get(char, word_dict["unk"]))
        dataset_x.append(sample_x)
        sample_x = []
    dataset_x = length_padding(dataset_x, word_dict)
    # print(len(dataset_x))
    return torch.LongTensor(dataset_x)


def length_padding(dataset_x, word_dict):
    max_length = 512
    for index in range(len(dataset_x)):
        if len(dataset_x[index]) > max_length:
            dataset_x[index] = dataset_x[index][:max_length]
        else:
            dataset_x[index] = dataset_x[index] + [word_dict["padding"]] * (max_length - len(dataset_x[index]))
    return dataset_x


def build_batch(train_sample_x, train_sample_y, batch, batch_size):
    return train_sample_x[batch * batch_size: (batch+1) * batch_size], train_sample_y[batch * batch_size: (batch+1) * batch_size]


def train(train_sample_x, train_sample_y, word_dict):
    epoch_num = 20
    word_dim = 768
    batch_size = 16
    category = 20
    max_length = 512
    total_sample = len(train_sample_x)
    batches = total_sample // batch_size
    model = MyModel(word_dim, category)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        model.to(device)
        for batch in range(batches):
            x, y = build_batch(train_sample_x, train_sample_y, batch, batch_size)
            # print(x, y)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = model(x, y)
            # print(type(loss), loss)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            print(loss)
        # logger.info("epoch average loss: %f" % np.mean(train_loss))
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, word_dict, max_length)  # 测试本轮模型结果


def evaluate(model, word_dict, max_length):
    model.eval()


def main():
    corpus_path = "tag_news.json"
    train_corpus_path = "train_tag_news.json"
    # valid_corpus_path = "valid_tag_news.json"
    word_dict = build_word_dict(corpus_path)
    train_sample_x, train_sample_y = build_sample(train_corpus_path, word_dict)
    train(train_sample_x, train_sample_y, word_dict)
    # print(word_dict)


if __name__ == "__main__":
    main()
