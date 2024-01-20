from loader import data_loader
import torch
import numpy as np
from torch.nn import functional


class Evaluate:
    def __init__(self, config, model, logger, cuda_flag):
        """
        接收main函数传递的三个参数，
        生成self.train_data，此数据库相当于实际问题中的faq库，用于相似度对比
        self.question_to_standard_question = {}记录self.questions = []中所有扩展问的索引到标准问索引的映射
        self.questions = []存储所有的扩展问，到标准问索引的映射记录在self.question_to_standard_question = {}中
        加载self.valid_data数据，
        :param config:
        :param model:
        :param logger:
        """
        super(Evaluate, self).__init__()
        self.config = config
        self.model = model
        self.logger = logger
        self.cuda_flag = cuda_flag
        self.train_data = data_loader(config["train_data_path"], config)
        self.valid_data = data_loader(config["valid_data_path"], config)
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.question_to_standard_question = {}
        self.questions = []

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            # print("batch_data:", batch_data)
            if self.cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            test_questions, labels = batch_data
            with torch.no_grad():
                test_question_vectors = self.model(test_questions)
            self.write_states(test_question_vectors, labels)
        self.show_states()
        return

    def knwb_to_vector(self):
        """

        :return: 生成一个包含所有扩展问的列表self.questions；
                生成了一个self.questions中所有扩展问到标准问索引字典的映射字典
        """
        for standard_question_index, expend_questions in self.train_data.dataset.knwb.items():
            for question in expend_questions:
                self.question_to_standard_question[len(self.questions)] = standard_question_index
                self.questions.append(question)
        with torch.no_grad():
            question_matrixs = torch.stack(self.questions, dim=0)
            # print("self.question_ids:", self.question_ids) 一个列表，每个问题是一个一维tensor保存在里面
            # print("question_matrixs:", question_matrixs)  将第一维度拼接得到一个二维的tensor
            if self.cuda_flag:
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            # question_matrixs只经过了编码环节，没有经过余弦相似度计算，所以没有归一化。将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def write_states(self, test_question_vectors, labels):
        assert len(test_question_vectors) == len(labels)
        for test_question_vector, label in zip(test_question_vectors, labels):
            test_question_vector = self.question_to_standard_question[test_question_vector]
            test_question_vector = torch.nn.functional.normalize(test_question_vector, dim=-1)  # 1 * vec_size
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            # 1 * vec_size  *  vec_size * len(self.knwb_vectors) => 1 * len(self.knwb_vectors)
            pre_index = int(torch.argmax(res.squeeze()))  # 命中问题标号
            pre_index = self.question_to_standard_question[pre_index]  # 转化成标准问编号
            if pre_index == label:
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_states(self):
        correct_num = self.stats_dict["correct"]
        wrong_num = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct_num + wrong_num))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct_num, wrong_num))
        self.logger.info("预测准确率：%f" % (correct_num / (correct_num + wrong_num)))
        self.logger.info("--------------------")
        return
