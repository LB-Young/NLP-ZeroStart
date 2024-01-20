# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
import json
from config import Config
from evaluate import Evaluator
from loader import load_data
from transformer.Models import Transformer
# 这个transformer是本文件夹下的代码，和我们之前用来调用bert的transformers第三方库是两回事

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
通过transformer模型进行文章标题的生成
"""

# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    # 输出模型的超参数配置

    model = Transformer(config["vocab_size"], config["vocab_size"], 0, 0,
                        d_word_vec=128, d_model=128, d_inner=256,
                        n_layers=1, n_head=2, d_k=64, d_v=64,
                        )
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 加载loss
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
    # 训练
    for epoch in range(config["epochs"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, input_position_seq, target_seq, tag_position_seq, gold = batch_data
            """
            input_seq, input_position_seq,          encoder部分需要的两个输入
            target_seq, tag_position_seq,           decoder部分需要的两个输入
            gold                                    decoder部分需要的一个输出
            """


            pred = model(input_seq, input_position_seq, target_seq, tag_position_seq)
            loss = loss_func(pred, gold.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss.append(float(loss))

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

        # if epoch % 20 == 0:
        #     model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        #     torch.save(model.state_dict(), model_path)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % config["epochs"])
    # torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
