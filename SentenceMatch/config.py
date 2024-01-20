Config = {
    "model_path": "model_output",
    "train_data_path": "data/train.json",
    "valid_data_path": "data/valid1.json",
    "schema_path": "data/schema.json",
    "vocab_path": "data/chars.txt",

    "class_method": "cosine_distance",
    "epochs": 10,
    "epoch_data_size": 30,
    "batch_size": 8,
    "max_length": 50,
    "learning_rate": 1e-5,
    "optimizer": "adam",
    "hidden_size": 256,
    "positive_sample_rate": 0.5,  # 正样本比例
    "pretrain_model_path": r"E:\python project\badou_new\bert-base-chinese"
}
