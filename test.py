import numpy as np
word_set = set()
with open("data/text8", "r") as f:
    for line in f:
        for word in line.split(" "):
            word_set.add(word)
print(word_set)
print(len(word_set))
