from pprint import pprint as pp
# coding:utf-8
import MeCab

sentence = "文書分類などでは、文書を単語の集合で表すことがよくあります。"

# MeCabの形態素解析器(Tagger)の初期化
tagger = MeCab.Tagger()
tagger.parse('')

# sentenceをTaggerによって形態素解析する
# 結果はNodeクラスのオブジェクトの単方向リストのルートノードとして返ってくる．
node = tagger.parseToNode(sentence)

print("=" * 50)
while node:
	# 形態素の表層形
	print(node.surface)

	# node.feature にはそれぞれの形態素の詳細がカンマ区切りで含まれている
	features = node.feature.split(",")
	features = [features[i] if len(features) > i else None for i in range(0, 9)]
	# features[0]: 品詞
	print("\t品詞\t: " + features[0])
	# features[1]: 品詞細分類1
	print("\t品詞細分類1\t: " + features[1])
	# features[2]: 品詞細分類2
	print("\t品詞細分類2\t: " + features[2])
	# features[3]: 品詞細分類3
	print("\t品詞細分類3\t: " + features[3])
	# features[4]: 活用形
	print("\t活用形\t: " + features[4])
	# features[5]: 活用型
	print("\t活用型\t: " + features[5])
	# features[6]: 原形
	print("\t原形\t: " + features[6])
	# features[7]: 読み
	print("\t読み\t: " + features[7])
	# features[8]: 発音
	print("\t発音\t: " + features[8])
	print("=" * 50)
	node = node.next
	pass