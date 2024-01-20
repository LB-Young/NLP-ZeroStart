import logging
import os
import torch
import numpy as np

from config import Config
from loader import data_loader
from model import MyModel, choose_optimizer
from evaluate import Evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(config):

    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])

    train_data = data_loader(config["train_data_path"], config)
    model = MyModel(config)
    optimizer = choose_optimizer(model, config)
    evaluator = Evaluate(model, config, logger)

    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()

    for epoch in range(config["epochs"]):
        logger.info("epoch %d begin" % epoch)
        model.train()
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            input_ids, labels = batch_data

            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if index % int(len(train_data)/2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
        if epoch % 20 == 0:
            model_path = os.path.join(config["model_path"], "num_layer_3_epoch_%d.pth" % epoch)
            torch.save(model.state_dict(), model_path)
    model_path = os.path.join(config["model_path"], "num_layer_3_epoch_%d.pth" % config["epochs"])
    torch.save(model.state_dict(), model_path)
    return


if __name__ == "__main__":
    main(Config)
