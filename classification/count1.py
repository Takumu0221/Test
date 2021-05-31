from pprint import pprint as pp
# coding:utf-8
import csv

# ファイルをオープン
with open('companies_tokenized1.csv') as f:
    reader = csv.reader(f)
    data = [x for x in reader]

# 全単語数を把握
words = []
for i in range(1, len(data)):
    # 単語ごとに分割
    sentence = data[i][4].split(' ')
    # print(sentence)

    # 単語集に追加
    for word in sentence:
        if word not in words:
            words.append(word)

print(words)
n = len(words)  # 総単語数

for i in range(len(data)):
    sentence = data[i][4].split(' ')
    # print(sentence)

    # 単語の数を計測
    word_num = [0] * n
    for word in sentence:
        if word in words:
            word_num[words.index(word)] += 1

    # dataに統合
    data[i] = data[i][0:4] + word_num

print(data)
# print(data[1])

# csvファイルに書き込み
with open("companies_bow1.csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerows(data)
