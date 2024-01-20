from model import MyModel
from config import Config
import numpy as np
from collections import defaultdict
import re
import json
import torch


class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config

        self.schema = load_schema(config["schema_path"])
        self.index_to_sign = dict((y, x) for x, y in self.schema.items())
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_len"] = len(self.vocab)

        self.model = MyModel(config)
        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        print("model加载完毕！")

    def predict(self, sent):
        input_ids = []
        for char in sent:
            input_ids.append(self.vocab.get(char, self.vocab["[UNK]"]))

        with torch.no_grad():
            predict = self.model(torch.LongTensor([input_ids]))[0]
        results = self.decode(sent, predict)
        return results

    def decode(self, sent, labels):
        labels = "".join([str(int(x)) for x in labels[:len(sent)]])  # re只对字符串作用，需要先将列表转换为字符串
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            # 取出匹配到的index <re.Match object; span=(0, 3), match='www'>
            results["LOCATION"].append(sent[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sent[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sent[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sent[s:e])
        return results


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


if __name__ == "__main__":
    # 300epoch左右的model文件识别效果较好，500左右也可以
    sl = SentenceLabel(Config, "model_output/epoch_500.pth")

    sentence = "中共中央政治局委员、书记处书记丁关根主持今天的座谈会"
    res = sl.predict(sentence)
    print(res)

    sentence = "延安是世界人民敬仰和向往的地方,曾接待大量外宾,周恩来指示招待外宾一定要体现艰苦奋斗的精神,要吃一点小米。"
    res = sl.predict(sentence)
    print(res)

    # sentence = "大屏幕涌现叶倩文年轻照，想起曾经过往，叶倩文哭的泣不成声！"
    # res = sl.predict(sentence)
    # print(res)
