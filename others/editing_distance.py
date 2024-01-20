import numpy as np


def editing_distance(str1, str2):
    row = len(str1) + 1
    col = len(str2) + 1
    res = np.zeros((row, col))
    print(len(res))
    for i in range(row):
        res[i][0] = i
    for j in range(col):
        res[0][j] = j
    print(res)
    for i in range(1, row):
        for j in range(1, col):
            if str1[i-1] == str2[j-1]:
                res[i][j] = min(res[i-1][j] + 1, res[i][j-1] + 1, res[i-1][j-1])
                '''
                res[i-1][j] + 1:表示从str1变到str2需要删除一个元素
                res[i][j-1] + 1:表示从str1变到str2需要增加一个元素
                res[i-1][j-1] + 1:表示从str1变到str2，如果相等则不需要元素变换，不等则需要一次元素变换
                '''
            else:
                res[i][j] = min(res[i-1][j] + 1, res[i][j-1] + 1, res[i-1][j-1] + 1)
            # if str1[i-1] == str2[j-1]:
            #     d = 0
            # else:
            #     d = 1
            # res[i][j] = min(res[i - 1][j] + 1, res[i][j - 1] + 1, res[i - 1][j - 1] + d)
    return res


if __name__ == "__main__":
    print(editing_distance("abcd", "acde"))
    # res = [[[0] * 3] * 3] * 3
    # print(res)
    # res[0][0][0] = 2
    # print(res)