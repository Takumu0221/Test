from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time

INPUT = 'companies_bow1.csv'

print("データセット : " + str(INPUT))
df = pd.read_csv(INPUT)

# 訓練データの準備
df_training = df[df['training/test'] == 'training']
X_train = df_training.drop(['index', 'company', 'category', 'training/test'], axis=1).values
y_train = df_training['category'].ravel()

# テストデータの準備
df_test = df[df['training/test'] == 'test']
X_test = df_test.drop(['index', 'company', 'category', 'training/test'], axis=1).values
y_test = df_test['category'].ravel()


def main():
    # SVM
    # 分類器の定義
    print("\n~SVM~")
    clf = SVC(random_state=1, C=10, gamma=0.01, kernel='rbf')
    classification(clf)

    # LinearSVM
    # 分類器の定義
    print("\n~Linear SVM~")
    clf = LinearSVC()
    classification(clf)

    # k近傍
    # 分類器の定義
    print("\n~K近傍法~")
    clf = KNeighborsClassifier()
    classification(clf)

    # 決定木
    # 分類器の定義
    print("\n~決定木~")
    clf = tree.DecisionTreeClassifier()
    classification(clf)

    # neural net
    print("\n~Neural Net~")
    clf = MLPClassifier()
    classification(clf)


def classification(clf):
    # 分類器の訓練
    t1 = time.time()
    clf = clf.fit(X_train, y_train)
    t2 = time.time()
    print("訓練にかかった時間 : {}".format(t2 - t1))

    # 訓練後の分類器による分類
    pred = clf.predict(X_test)

    # 正解(y_test)と分類結果(pred)の一致度(=正解率)を計算して出力
    accuracy_test = accuracy_score(y_test, pred)
    print('テストデータに対する正解率： %.4f' % accuracy_test)

    # 混同行列を出力(45~54行目は、綺麗に出力するために冗長になっているのであまり理解しなくていいです)
    labels = list(set(df['category']))
    conf_matrix = confusion_matrix(y_test, pred, labels=labels)

    print("\n【分類結果の詳細】")
    for label in labels:
        print('\t' + label, end='')
    print()

    for conf, label in zip(conf_matrix, labels):
        print(label + '\t', end='')
        for c in conf:
            print(str(c) + '\t', end='')
        print('')


if __name__ == '__main__':
    main()
