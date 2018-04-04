# transform glove text into glove pkl
import numpy as np
import pickle as pkl


"""
2196017 300
"""
flag = 0
glove_dict = dict()
with open("../GloVe.txt", "r") as f:
    for line in f:
        line = line.strip().split(" ")
        if flag == 0:
            flag += 1
            continue
        tmp_vec_ = np.array(line[1:])
        tmp_vec = tmp_vec_.astype(np.float64)
        glove_dict[line[0]] = tmp_vec
print("包含的单词个数", len(glove_dict.keys()))
# 存储
with open("../glove.pkl", "wb") as f:
    pkl.dump(glove_dict, f)
print("存储完毕")

# arr_ = np.array(["2.1", "23.3"])
# arr = arr_.astype(np.float64)
# print(arr)
# print(type(arr[0]))
