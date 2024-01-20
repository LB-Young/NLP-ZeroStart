# coding:utf8

Config = {
    "data_path": "../tag_news",
    "train_data_path": "../train_tag_news.json",
    "valid_data_path": "../valid_tag_news.json",
    "chars_path": "../chars.txt",
    "pooling_style": "max",

    "batch_size": 16,
    "optimizer": "Adam",
    "learn_rate": 1e-5,

    "kernel_size": 3,

    "max_length": 30,
    "embedding_size": 768,
    "hidden_size": 128,
    "layer_num": 1,         # LSTM,RNN,GRU
    "model_type": "lstm",
    "model_path": "result",
    "pretrain_model_path": r"E:\python project\badou_new\bert-base-chinese",
    "epochs": 1,
    "random_seed": 123,

    "vocab_size": "",
    "class_num": 18
}