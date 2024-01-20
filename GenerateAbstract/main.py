# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import logging
import json
from config import Config
from evaluate import Evaluator
from loader import load_data
from Models import Transformer
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

    model = Transformer(src_vocab_size=config["vocab_size"],
                        tag_vocab_size=config["vocab_size"],
                        position_num=config["input_max_length"],
                        src_padding_index=0,
                        tag_padding_index=0,
                        embedding_size=128,
                        block_num=1,
                        head_num=2,
                        model_size=128,
                        forward_hidden_size=512,
                        kq_d=64,
                        v_d=64,
                        dropout=0.5,
                        enc_dec_embedding_share=True,
                        dec_out_embedding_share=True
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

            optimizer.zero_grad()
            pred = model(input_seq, input_position_seq, target_seq, tag_position_seq)
            loss = loss_func(pred, gold.view(-1))
            loss.backward()
            optimizer.step()

            train_loss.append(float(loss))

        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)

        # if epoch % 20 == 0:
        #     model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
        #     torch.save(model.state_dict(), model_path)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % config["epochs"])
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
