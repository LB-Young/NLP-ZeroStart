from loader import data_loader
import torch
import json
import re
import numpy as np
from collections import defaultdict


class Evaluate:
    def __init__(self, model, config, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = data_loader(self.config["valid_data_path"], config)
        self.schema = schema_loader(self.config["schema_path"])
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()

    def eval(self, epoch):
        self.logger.info("当前epoch：%d预测结果" % epoch)
        for index, batch_data in enumerate(self.valid_data):
            sentences = self.valid_data.dataset[index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data
            with torch.no_grad():
                predicts = self.model(input_ids)
            self.write_state(predicts, labels, sentences)
        self.show_state()
        return

    def write_state(self, predicts, labels, sentences):
        # print(len(predicts), len(labels), len(sentences))
        assert len(predicts) == len(labels) == len(sentences)

        if not self.config["use_crf"]:
            predicts = torch.argmax(predicts, dim=-1)

        for predict, label, sentence in zip(predicts, labels, sentences):
            if not self.config["use_crf"]:
                predict = predict.cpu().detach().tolist()
            label = label.cpu().detach().tolist()

            pred_entities = self.decode(predict, sentence)
            true_entities = self.decode(label, sentence)
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len([ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def decode(self, label, sentence):
        label = "".join(str(char) for char in label[: len(sentence)])
        results = defaultdict(list)
        for location in re.finditer("04+", label):
            start, end = location.span()
            results["LOCATION"].append(sentence[start: end])
        for location in re.finditer("15+", label):
            start, end = location.span()
            results["ORGANIZATION"].append(sentence[start:end])
        for location in re.finditer("26+", label):
            start, end = location.span()
            results["PERSON"].append(sentence[start:end])
        for location in re.finditer("37+", label):
            start, end = location.span()
            results["TIME"].append(sentence[start:end])
        return results

    def show_state(self):
        f1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            f1 = (2 * recall * precision) / (recall + precision + 1e-5)
            f1_scores.append(f1)
        self.logger.info("macro f1: %f" % np.mean(f1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-F1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

def schema_loader(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)
