import os
import torch
import numpy as np
import random
import csv
import logging
from config import Config
from loader import load_data
from model import MyModel, ChooseOptimizer
from evaluate import Evaluate

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config):
    train_data = load_data(config["train_data_path"], config)      # len(train_data) 90 分成了90个batch
    model = MyModel(config)
    optimizer = ChooseOptimizer(config, model)

    # model.to(device)
    model = model.cuda()
    evaluator = Evaluate(config, model, logger)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            # print("batch_data:", batch_data)
            # batch_data.to(device)
            batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            data_x2id, labels = batch_data
            # data_x2id: torch.Size([16, 512]) labels: torch.Size([16, 1])
            # print('data_x2id:', data_x2id.shape, data_x2id.type, "labels:", labels.shape, labels.type)
            # data_x2id: torch.Size([16, 512]) <built-in method type of Tensor object at 0x0000028C4718A368>
            # labels: torch.Size([16, 1]) <built-in method type of Tensor object at 0x0000028C4718A9A8>
            loss = model(data_x2id, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss)

            if index % int(len(train_data) / 2) == 0:       # print twice every epoch
                logger.info("epoch: %d; batch_loss %f" % (epoch, loss))
    acc = evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"],
    #                           "%s__lr=%.3f__hs=%d__bs=%d__pool=%s__epoch=%d__acc=%.2f%%.pth" % (
    #                               config["model_type"], config["learn_rate"],
    #                               config["hidden_size"], config["batch_size"],
    #                               config["pooling_style"], epoch, acc * 100))
    # torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc, loss


def csv_writer(data):
    """
    将结果写入csv文件
    """
    a = []
    dict = data[0]
    for headers in dict.keys():  # 把字典的键取出来
        a.append(headers)
    header = a  # 把列名给提取出来，用列表形式呈现
    # a表示以“追加”的形式写入,“w”的话，表示在写入之前会清空原文件中的数据
    # newline是数据之间不加空行
    with open('result.csv', 'a', newline='', encoding='utf-8') as f:
        # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        writer = csv.DictWriter(f, fieldnames=header)
        # 写入列名
        writer.writeheader()
        # 写入数据
        writer.writerows(data)
    print("数据已经写入成功！！！")


def compare_parameters(Config):
    result_list = []
    for model in ["bert", "lstm", "gated_cnn", "rcnn", "bert_lstm", "bert_cnn"]:
        Config["model_type"] = model
        for lr in [1e-4, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [16]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", "max"]:
                        Config["pooling_style"] = pooling_style
                        Config["acc"], Config["loss"] = main(Config)  # 记录最后一轮的准确率
                        dict_temp = {"model_type": Config['model_type'],
                                     "max_length": Config['max_length'],
                                     "hidden_size": Config['hidden_size'],
                                     "kernel_size": Config['kernel_size'],
                                     "layer_num": Config['layer_num'],
                                     "epochs": Config['epochs'],
                                     "batch_size": Config['batch_size'],
                                     "pooling_style": Config['pooling_style'],
                                     "optimizer": Config['optimizer'],
                                     "learn_rate": Config['learn_rate'],
                                     "random_seed": Config['random_seed'],
                                     "loss": Config["loss"],
                                     "acc": Config['acc']}
                        # print("当前配置：\n", Config)
                        result_list.append(dict_temp)
    csv_writer(result_list)


if __name__ == "__main__":
    acc, loss = main(Config)
    print("acc:", acc)
    print("loss:", loss)
    # compare_parameters(Config)
