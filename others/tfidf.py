import math
import os
from collections import defaultdict

"""
tfidf的计算和使用
"""

# 统计tf和idf值


def build_tf_idf_dict(corpus):
    """

    :param corpus: 分词之后的corpus，是个列表，且每个元素是一个列表（一个文件切分之后得到的结果列表）
    :return:
    """
    # print(corpus)
    tf_dict = defaultdict(dict)  # key:文档序号，value：dict，文档中每个词出现的频率
    idf_dict = defaultdict(set)  # key:词， value：set，文档序号，最终用于计算每个词在多少篇文档中出现过
    # idf_dict用set而不用list是因为某个词可以在一篇文章中出现多次，如果用list会有元素重复
    for text_index, text_words in enumerate(corpus):
        # print(text_index, text_words)
        for word in text_words:
            if word not in tf_dict[text_index]:
                tf_dict[text_index][word] = 0
            tf_dict[text_index][word] += 1
            idf_dict[word].add(text_index)
    idf_dict = dict([(key, len(value)) for key, value in idf_dict.items()])
    return tf_dict, idf_dict

# 根据tf值和idf值计算tfidf


def calculate_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = defaultdict(dict)
    for text_index, word_tf_count_dict in tf_dict.items():
        for word, tf_count in word_tf_count_dict.items():
            tf = tf_count / sum(word_tf_count_dict.values())
            # tf-idf = tf * log(D/(idf + 1))
            tf_idf_dict[text_index][word] = tf * math.log(len(tf_dict)/(idf_dict[word]+1))
    return tf_idf_dict

# 输入语料 list of string
# ["xxxxxxxxx", "xxxxxxxxxxxxxxxx", "xxxxxxxx"]


def calculate_tfidf(corpus):
    """

    :param corpus: 输入的文本内容，一个文件为一个元素
    :return:
    """
    # 先进行分词
    corpus = [text.split() for text in corpus]
    # print(corpus[0])
    tf_dict, idf_dict = build_tf_idf_dict(corpus)
    tf_idf_dict = calculate_tf_idf(tf_dict, idf_dict)
    return tf_idf_dict

# 根据tfidf字典，显示每个领域topK的关键词


def tf_idf_topk(tfidf_dict, paths=[], top=10, print_word=True):
    topk_dict = {}
    for text_index, text_tfidf_dict in tfidf_dict.items():
        word_list = sorted(text_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        # 参数key=lambda x: x[1]表示用value值排序，key=lambda x: x[0]表示用keys值排序
        print(len(word_list))
        topk_dict[text_index] = word_list[:top]
        if print_word:
            print(text_index, paths[text_index])
            for i in range(top):
                print(word_list[i])
            print("----------")
    return topk_dict


def main():
    dir_path = r"category_corpus/"
    corpus = []
    paths = []
    for path in os.listdir(dir_path):
        path = os.path.join(dir_path, path)
        if path.endswith("txt"):
            corpus.append(open(path, encoding="utf8").read())
            paths.append(os.path.basename(path))
            # 将所有文件内容都合并到corpus列表中每个文件是一个元素，将所有文件名合并到paths列表中
            # print(len(corpus))
            # print(paths)
    tf_idf_dict = calculate_tfidf(corpus)
    print(tf_idf_dict)
    tf_idf_topk(tf_idf_dict, paths)


if __name__ == "__main__":
    main()