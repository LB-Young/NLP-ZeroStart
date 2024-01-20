import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torchcrf import CRF


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        vocab_len = self.config["vocab_len"]
        embedding_size = self.config["embedding_size"]
        hidden_size = self.config["hidden_size"]
        class_num = self.config["class_num"]
        self.embedding = nn.Embedding(vocab_len, embedding_size, padding_idx=0)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True, num_layers=self.config["num_of_layers"])
        self.classifier = nn.Linear(hidden_size * 2, class_num)
        self.use_crf = self.config["use_crf"]
        self.crf = CRF(class_num, batch_first=True)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input, label=None):
        x = self.embedding(input)
        print(x.shape)
        x, _ = self.rnn(x)
        predict = self.classifier(x)
        if label is not None:
            if self.use_crf:
                # print("label", label)
                mask = label.gt(-1)
                # print("mask", mask)
                return - self.crf(predict, label, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), label.view(-1))
        else:
            if self.use_crf:
                return self.crf.decode(predict)
            else:
                return predict


def choose_optimizer(model, config):
    optim = config["choose_optimizer"]
    lr = config["learning_rate"]
    if optim == "adam":
        return Adam(model.parameters(), lr=lr)
    else:
        return SGD(model.parameters(), lr=lr)