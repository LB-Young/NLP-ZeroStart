import os
import logging
import torch
import numpy as np
from config import Config
from model import MyModel, choose_optimizer
from loader import data_loader
from evaluate import Evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = data_loader(config["train_data_path"], config)
    model = MyModel(config)
    optimizer = choose_optimizer(config, model)

    # cuda_flag = torch.cuda.is_available()
    cuda_flag = False
    if cuda_flag:
        model = model.cuda()

    evaluator = Evaluate(config, model, logger, cuda_flag)

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_ids1, input_ids2, label = batch_data

            optimizer.zero_grad()
            loss = model(input_ids1, input_ids2, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss:%f" % np.mean(train_loss))
        evaluator.eval(epoch)
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % config["epochs"])
    model.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
