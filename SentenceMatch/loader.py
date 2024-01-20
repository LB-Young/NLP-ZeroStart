import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from transformers import BertTokenizer
import json
import jieba
import random




class DataGenerator:
    def __init__(self, data_path, config):
        super(DataGenerator, self).__init__()
        self.path = data_path
        self.config = config
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        # print("self.schema:", self.schema)
        self.train_data_path = config["train_data_path"]
        self.max_length = config["max_length"]
        self.data_type = None
        self.data = []
        self.knwb = defaultdict(list)
        self.load()

    def __len__(self):
        return self.config['train_steps']

    def single_sample(self):
        """
        self.question_data:
        {"question": "你知道有哪些企业生产空气净化机吗",
         "table_id": "a7b5f29c3b0611e9908bf40f24344a08",
         "sql": {"agg": [0], "cond_conn_op": 0, "sel": [1], "conds": [[3, 2, "空气净化机"]]}}

         self.table_data： dict格式
         每个元素的key是ids，value是一个list：里面保存了原表，转置表，去重表，str类型行， number类型行
        """
        question_data = random.choice(self.question_data)
        table_data = self.table_data[question_data["table_id"]]
        all_header_indexs = list(range(len(table_data['header'])))
        select_header_indexs = question_data['sql']['sel']
        agg_header_indexs = question_data['sql']['agg']
        conditions = question_data['sql']['conds']
        where_header_indexs = [x[0] for x in conditions]

        if random.random() > pos_sample_rate:
            sample_header_index = random.choice(all_header_indexs)
        else:
            sample_header_index = random.choice(select_header_indexs + where_header_indexs)

        if sample_header_index in where_header_indexs:
            label = 6
        elif sample_header_index in select_header_indexs:
            label = agg_header_indexs[select_header_indexs.index(sample_header_index)]
        else:
            label = 7

        text = question_data['question']
        head = table_data['header'][sample_header_index]

        encode = self.tokenizer.encode_plus(head,
                                            text,
                                            truncation=True,
                                            max_length=self.config["max_length"],
                                            padding='max_length'
                                            )
        # print(encode)
        input_ids = encode["input_ids"]
        attention_mask = encode["attention_mask"]
        token_type_ids = encode["token_type_ids"]

        return input_ids, attention_mask, token_type_ids, label

    def __iter__(self):
        steps = self.config["train_steps"]
        batch_size = self.config["batch_size"]
        for i in range(steps):
            input_ids_list, attention_mask_list, token_type_list, labels_list = [], [], [], []
            for _ in range(batch_size):
                input_ids, attention_mask, token_type, label = self.single_sample()
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                token_type_list.append(token_type)
                labels_list.append(label)
            input_ids_list = torch.LongTensor(input_ids_list)
            attention_mask_list = torch.LongTensor(attention_mask_list)
            token_type_list = torch.LongTensor(token_type_list)
            labels_list = torch.LongTensor(labels_list)
            yield input_ids_list, attention_mask_list, token_type_list, labels_list

    def load(self):
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                if isinstance(line, dict):
                    self.data_type = "train"
                    target = line["target"]
                    target_label = self.schema[target]
                    # print("target_label:", target_label)
                    for question in line["questions"]:
                        input_ids = self.encode_sentence(question)
                        input_ids = torch.LongTensor(input_ids)
                        # print("input_ids:", input_ids)
                        self.knwb[target_label].append(input_ids)
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    target = line[1]
                    question = line[0]
                    # print(self.schema[target])
                    target_index = torch.LongTensor([self.schema[target]])
                    # print("target_index:", target_index)
                    input_ids = self.encode_sentence(question)
                    input_ids = torch.LongTensor(input_ids)
                    # print("input_ids:", input_ids)
                    self.data.append([input_ids, target_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            text = jieba.lcut(text)
            for word in text:
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.max_length]
        input_id += [0] * (self.max_length - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    def random_train_sample(self):
        standard_questions_index = list(self.knwb.keys())
        if random.random() < self.config["positive_sample_rate"]:
            index = random.choice(standard_questions_index)
            if len(self.knwb[index]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.knwb[index], 2)
                return [s1, s2, torch.LongTensor([1])]
        else:
            index1, index2 = random.sample(standard_questions_index, 2)
            s1 = random.choice(self.knwb[index1])
            s2 = random.choice(self.knwb[index2])
            return [s1, s2, torch.LongTensor([-1])]


def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())


def data_loader(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl