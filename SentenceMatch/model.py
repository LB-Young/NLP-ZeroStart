import torch
import torch.nn as nn
from torch.nn import functional
from torch.optim import Adam, SGD
from transformers import BertModel


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        pretrain_model_path = config["pretrain_model_path"]
        self.bert_encoder = BertModel.from_pretrained(pretrain_model_path)
        self.bert_hidden_size = self.bert_encoder.pooler.dense.out_features
        self.encode_out = nn.Linear(self.bert_hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.bert_encoder(x)[1]
        x_encode = self.encode_out(x)
        return x_encode


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.encoder = BertEncoder(config)
        hidden_size = self.encoder.bert_hidden_size
        self.classifier = nn.Linear(hidden_size * 2, 2)
        self.loss = nn.CosineEmbeddingLoss()

    def forward(self, input_ids1, input_ids2=None, label=None):
        if self.config["class_method"] == "cosine_distance":
            if input_ids2 is not None:
                vector1 = self.encoder(input_ids1)
                vector2 = self.encoder(input_ids2)
                if label is not None:
                    return self.loss(vector1, vector2, label.squeeze())
                else:
                    return self.cosine_distance(vector1, vector2)
            else:
                return self.encoder(input_ids1)
        else:
            if input_ids2 is not None:
                vector1 = self.encoder(input_ids1)
                vector2 = self.encoder(input_ids2)
                vecter = torch.concat([vector1, vector2], dim=1)
                out = self.classifier(vecter)
                if label is not None:
                    return self.loss(out, label.squeeze())
                else:
                    return out
            else:
                return self.self.encoder(input_ids1)

    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine_dis = torch.sum(torch.mul(tensor1, tensor2), dim=1)
        return 0.5 * (1 + cosine_dis)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    lr = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=lr)
    else:
        return SGD(model.parameters(), lr=lr)
