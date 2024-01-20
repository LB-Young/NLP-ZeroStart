import torch
import torch.nn as nn
from torch.nn import functional
from torch.optim import Adam, SGD
from transformers import BertModel


class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        hidden_size = config["hidden_size"]
        embedding_size = config["embedding_size"]
        layer_num = config["layer_num"]
        class_num = config["class_num"]
        vocab_size = config["vocab_size"]       # vocab_size: 4624
        model_type = config["model_type"]
        pretrain_model_path = config["pretrain_model_path"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)   # 4624 * 128
        self.use_bert = False
        if model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(pretrain_model_path)
            self.encoder.config.return_dict = False
            hidden_size = self.encoder.config.hidden_size       # 取出encoder的隐层size: 784
            # Bert与其他模型embedding之后，输出维度不一样
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=layer_num)
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=layer_num)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=layer_num)
        elif model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "text_cnn":
            self.encoder = TextCNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
        else:
            self.encoder = BertMidLayer(config)

        self.classify = nn.Linear(hidden_size, class_num)
        self.classify1 = nn.Linear(embedding_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = functional.cross_entropy

    def forward(self, x, y=None):
        print(x.shape, x.shape)
        # <built-in method type of Tensor object at 0x000002392A471818> torch.Size([16, 512])
        if self.use_bert:
            x = self.encoder(x)
            # x = x.last_hidden_state
        else:
            x = self.embedding(x)
            print(x.shape, x.shape)
            x = self.encoder(x)

        # print("输入x:", x)
        if isinstance(x, tuple):    # 如果返回值是tuple类型，则使用的模型是Bert，rnn，lstm这些，我们需要取出第一维度
            x = x[0]
        # print(x)
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()

        # print("predict:", predict)        # batch_size * max_length
        # print("y:", y)                    # batch_size * 1
        if len(x[0]) == 768:        # bert输出与无Bert模型的输出维度不同，固分类器的输入维度也应该不同
            predict = self.classify1(x)
        else:
            predict = self.classify(x)

        if y is not None:
            # print("predict.shape:", predict.shape)
            # print("y.squeeze().shape:", y.squeeze().shape)
            return self.loss(predict, y.squeeze())
        else:
            return predict


class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, bias=False, padding=pad)

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)
# TextCNN和TextCNN1要写两个，因为RNN和Bert的embedding维度不同


class TextCNN1(nn.Module):
    def __init__(self, config):
        super(TextCNN1, self).__init__()
        hidden_size = config["hidden_size"]
        embedding_size = config["embedding_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(embedding_size, hidden_size, kernel_size=kernel_size, bias=False, padding=pad)

    def forward(self, x):
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN1(nn.Module):
    def __init__(self, config):
        super(GatedCNN1, self).__init__()
        self.cnn = TextCNN1(config)
        self.gate = TextCNN1(config)

    def forward(self, x):
        out = self.cnn(x)
        gate = self.cnn(x)
        gate = torch.sigmoid(gate)
        return torch.mul(out, gate)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = TextCNN(config)
        self.gate = TextCNN(config)

    def forward(self, x):
        out = self.cnn(x)
        gate = self.cnn(x)
        gate = torch.sigmoid(gate)
        return torch.mul(out, gate)


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        hidden_out, _ = self.rnn(x)
        out = self.cnn(x)
        return out


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        # self.bert.config.return_dict = False
        self.cnn = GatedCNN1(config)

    def forward(self, x):
        hidden_out = self.bert(x).last_hidden_state
        # print(hidden_out)
        out = self.cnn(hidden_out)
        return out


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, bias=False, batch_first=True)

    def forward(self, x):
        # hidden_out = self.bert(x)[0]
        hidden_out = self.bert(x).last_hidden_state
        out, _ = self.lstm(hidden_out)      # torch.Size([16, 30, 768])
        return out


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        pretrain_model_path = config["pretrain_model_path"]
        self.bert = BertModel.from_pretrained(pretrain_model_path, output_hidden_states=True)
        # self.bert.config.return_dict = False

    def forward(self, x):
        # print(self.bert(x))
        hidden_states = self.bert(x).hidden_states
        # hidden_states = self.bert(x)[2]
        # _, _, hidden_states = self.bert(x)
        layers_states = torch.add(hidden_states[-1], hidden_states[-2])
        return layers_states


def ChooseOptimizer(config, model):
    optimizer = config["optimizer"]
    learn_rate = config["learn_rate"]
    if optimizer == "Adam":
        return Adam(model.parameters(), lr=learn_rate)
    elif optimizer == "SGD":
        return SGD(model.parameters(), lr=learn_rate)