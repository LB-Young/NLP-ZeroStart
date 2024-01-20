import re
import json
import pandas as pd
import itertools
from py2neo import Graph
from collections import defaultdict


class GraphQA:
    def __init__(self):
        self.graph = Graph("http://localhost:7474", auth=("neo4j", "122455"))
        self.question_templets = self.load_question_templet("question_templet.xlsx")
        self.entities, self.relations, self.labels, self.attributes = self.load_kg_schema("kg_schema.json")
        print("知识图谱数据和模板加载完毕！")

    def load_question_templet(self, path):
        """
        将"question_templet.xlsx"中模板按行转为list保存在一个列表中
        :param path: 模板文件路径
        :return:
        """
        dataframe = pd.read_excel(path)
        question_templets = []
        for i in range(len(dataframe)):
            question_pattern = dataframe["question"][i]
            cypher = dataframe["cypher"][i]
            answer = dataframe["answer"][i]
            check = dataframe["check"][i]
            question_templets.append([question_pattern, cypher, answer, json.loads(check)])
        return question_templets

    def load_kg_schema(self, path):
        """
        将kg_schema文件中的entity, relation, label, attribute加载好后面用于对输入问题进行基于正则的实体挖掘
        :param path: kg_schema路径
        :return:
        """
        with open(path, encoding="utf8") as f:
            kg_schema = json.load(f)
            entities = kg_schema["entitys"]
            relations = kg_schema["relations"]
            labels = kg_schema["labels"]
            attributes = kg_schema["attributes"]
            return entities, relations, labels, attributes

    def get_info(self, sentence):
        entities = re.findall("|".join(self.entities), sentence)
        relations = re.findall("|".join(self.relations), sentence)
        labels = re.findall("|".join(self.labels), sentence)
        attributes = re.findall("|".join(self.attributes), sentence)
        return {
            "%ENT%": entities,
            "%REL%": relations,
            "%LAB%": labels,
            "%ATT%": attributes
        }

    def expend_templet(self, info):
        expend_templets = []
        for question_templet in self.question_templets:
            # print("question_templet:", question_templet)
            templet, cypher, answer, check = question_templet
            if self.check_info(check, info):
                expend_templet = self.expend(question_templet, info)
                """
                expend_templet:将info排列组合填充到当前这个question_templet,得到的list
                """
                expend_templets += expend_templet
            else:
                continue
        # print("1111expend_templets:", expend_templets)
        return expend_templets

    def expend(self, question_templet, info):
        """
        先将 info 排列组合，然后按照排列组合的结果对 question_templet 内容进行填充
        :param question_templet:
        :param info:
        :return:
        """
        replaced_templet = []
        templet, cypher, answer, check = question_templet
        pailiezuhe_info = self.pailiezuhe(check, info)
        for zuhe_info in pailiezuhe_info:
            for key, val in zuhe_info.items():
                templet = re.sub(key, val, templet)
                cypher = re.sub(key, val, cypher)
                answer = re.sub(key, val, answer)
            replaced_templet.append([templet, cypher, answer])
        return replaced_templet

    def pailiezuhe(self, check, info):
        slot_values = []
        for key, required_count in check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, check))
        return combinations

    def decode_value_combination(self, value_combination, check):
        res = {}
        for index, (key, required_count) in enumerate(check.items()):
            if required_count == 1:
                res[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
        return res

    def check_info(self, check, info):
        # print("111111111111")
        for key, required_count in check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    def get_scores(self, sentence, expend_templets):
        """
        根据排列组合和填充完的 expend_templets 和 sentence 计算文本匹配得分
        :param sentence:
        :param expend_templets:
        :return:
        """
        for index, expend_templet in enumerate(expend_templets):
            score = self.cal_score(sentence, expend_templet)
            expend_templets[index].append(score)
        expend_templets = sorted(expend_templets, reverse=True, key=lambda x: x[-1])
        return expend_templets

    def cal_score(self, sentence, expend_templet):
        templet = expend_templet[0]
        score = len(set(templet) & set(sentence)) / len(set(templet) | set(sentence))
        return score

    def parse_result(self, graph_search_result, answer):
        # print("graph_search_result:", graph_search_result)
        graph_search_result = graph_search_result[0]
        # 关系查找返回的结果形式较为特殊，单独处理
        if "REL" in graph_search_result:
            # print("graph_search_result[REL]:", graph_search_result["REL"])
            # print("graph_search_result[REL].types():", graph_search_result["REL"].types())
            # print("list(graph_search_result[REL].types()):", list(graph_search_result["REL"].types()))
            # print("list(graph_search_result[REL].types())[0]:", list(graph_search_result["REL"].types())[0])
            graph_search_result["REL"] = list(graph_search_result["REL"].types())[0]
        answer = self.replace_token_in_string(answer, graph_search_result)
        return answer

    def replace_token_in_string(self, string, combination):
        # print("combination:", combination)
        for key, value in combination.items():
            # print(key)
            # print(value)
            string = re.sub(key, value, string)
        return string

    def query(self, sentence):
        """
        1、从sentence中通过正则匹配的方式获取实体保存在info中
        2、根据info内容对模板文件xlsx文件进行填充
            对Templeton中所有模板进行遍历，对符合check的模板取出来
                根据check和cypher内容对info进行排列组合，然后排列组合的结果对question、cypher、answer进行填充，将填充结果保存到list中返回query函数
        3、将输入问题和填充完之后的问题计算匹配得分
        4、按分数从高向低运行cypher，知道运行出结果

        :param sentence: 需要回答的问题
        :return:
        """
        info = self.get_info(sentence)
        # print("info:", info)
        expend_templets = self.expend_templet(info)
        # print("1111expend_templets:", expend_templets)
        templet_cypher_scores_answer = self.get_scores(sentence, expend_templets)
        # print("templet_cypher_scores_answer:", templet_cypher_scores_answer)
        for templet, cypher, answer, scores in templet_cypher_scores_answer:
            graph_search_result = self.graph.run(cypher).data()
            if graph_search_result:
                answer = self.parse_result(graph_search_result, answer)
                return answer
        return "None"


def main():
    graph = GraphQA()
    res = graph.query("谁导演的不能说的秘密")
    print(res)
    res = graph.query("发如雪的谱曲是谁")
    print(res)
    res = graph.query("爱在西元前的谱曲是谁")
    print(res)
    res = graph.query("周杰伦的星座是什么")
    print(res)
    res = graph.query("周杰伦的血型是什么")
    print(res)
    res = graph.query("周杰伦的身高是什么")
    print(res)
    res = graph.query("周杰伦和淡江中学是什么关系")
    print(res)


if __name__ == "__main__":
    main()
