import torch
from loader import load_data

class Evaluate():
    def __init__(self, config, model, logger):
        super(Evaluate, self).__init__()
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct": 0, "wrong": 0}  # 用于存储测试结果

    def eval(self, epoch):
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            batch_data = [d.cuda() for d in batch_data]
            data_x2id, label = batch_data
            with torch.no_grad():
                pred_result = self.model(data_x2id)
            # print("pred_result", pred_result)
            self.write_states(label, pred_result)
        acc = self.show_states()
        return acc

    def write_states(self, label, pred_result):
        assert len(label) == len(pred_result)
        for true_label, pred_label in zip(label, pred_result):
            # print("11", pred_label)
            pred_label = torch.argmax(pred_label)
            # print("11", true_label)
            # print("22", pred_label)
            # print("33", true_label, pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_states(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct + wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return correct / (correct + wrong)
