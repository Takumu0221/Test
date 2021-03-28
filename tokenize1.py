from pprint import pprint as pp
# coding:utf-8
import MeCab
import csv

with open('companies.csv') as f:
	reader = csv.reader(f)
	data = [x for x in reader]

for i in range(len(data)):
	sentence = data[i][4]
	# print(sentence)

	# MeCabの形態素解析器(Tagger)の初期化
	tagger = MeCab.Tagger()
	tagger.parse('')

	# sentenceをTaggerによって形態素解析する
	# 結果はNodeクラスのオブジェクトの単方向リストのルートノードとして返ってくる．
	node = tagger.parseToNode(sentence)

	sentence = ""

	while node:
		# 形態素の表層形
		# print(node.surface)

		# 単語分割後の文作成
		if node.surface != "":
			sentence += node.surface + " "

		# node.feature にはそれぞれの形態素の詳細がカンマ区切りで含まれている
		# features = node.feature.split(",")
		# print(features)
		# features = [features[i] if len(features) > i else None for i in range(0, 9)]

		node = node.next
		pass

	# dataに統合
	data[i][4] = sentence

# print(data[1])

# csvファイルに書き込み
with open("companies_tokenized1.csv", 'w') as f:
	writer = csv.writer(f)
	writer.writerows(data)
