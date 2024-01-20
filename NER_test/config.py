"""
配置参数
"""

Config = {
    "model_path": "model_output",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    "schema_path": "ner_data/schema.json",
    "vocab_path": "chars.txt",

    "use_crf": True,
    "embedding_size": 256,
    "hidden_size": 128,
    "class_num": 9,
    "num_of_layers": 1,
    "batch_size": 16,
    "max_length": 30,

    "epochs": 500,
    "learning_rate": 1e-3,
    "choose_optimizer": "adam"
}