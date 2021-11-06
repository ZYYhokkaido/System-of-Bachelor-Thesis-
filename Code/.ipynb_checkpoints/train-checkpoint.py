# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import svm, metrics, preprocessing #機械学習用のライブラリを利用
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from mlxtend.plotting import plot_decision_regions #学習結果をプロットする外部ライブラリを利用
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from preprocessing import preprocess_dataset
from utils import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import pickle

filename='models/LR(1:-1_5:1).sav'

# 2：Wineのデータセットを読み込む--------------------------------
# df_wine_all=pd.read_csv('data/amazon_reviews_multilingual_JP_v1_00.tsv',delimiter='\t',header=None)
# df_wine_all=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#品種(0列、1～3)と色（10列）とプロリンの量(13列)を使用する
# df_wine=df_wine_all[[7,12]]
# df_wine.columns = [u'star_rating', u'review_body']
# pd.DataFrame(df_wine)  #この行を実行するとデータが見れる


# 3：プロットしてみる------------------------------------------------------
# %matplotlib inline
# x=df_wine["color"]
# y=df_wine["proline"]
# z=df_wine["star_rating"]
#plt.scatter(x,y, c=z)
#plt.show

# 4：データの整形-------------------------------------------------------

x,y=load_dataset('data/amazon_reviews_multilingual_JP_v1_00_final.tsv')	
x=preprocess_dataset(x)

from janome.tokenizer import Tokenizer
t=Tokenizer(wakati=True)

vectorizer=TfidfVectorizer(tokenizer=t.tokenize)
tfidf=vectorizer.fit_transform(x)
vocab=vectorizer.get_feature_names()
x=pd.DataFrame(tfidf.toarray(),columns=vocab).to_numpy()
print(x.shape)

#解説 5：機械学習で分類する---------------------------------------------------
lr=LogisticRegression()#loss='squared_hinge' #loss="hinge", loss="log"

# # 6：K分割交差検証（cross validation）で性能を評価する---------------------
# scores=cross_val_score(lr, x, y, cv=10)
# print("平均正解率 = ", scores.mean())
# print("正解率の標準偏差 = ", scores.std())

# 7：トレーニングデータとテストデータに分けて実行してみる------------------
X_train, X_test, Y_train, Y_test=train_test_split(x,y, test_size=0.2, random_state=0)
# lr.fit(X_train, Y_train)

# pickle.dump(lr, open(filename, 'wb'))

lr = pickle.load(open(filename, 'rb'))



#正答率を求める
""" Y_pred=lr.predict(X_test)
X_pred=lr.predict(X_train)

print('train:')
print('confusion matrix = \n', confusion_matrix(y_true=Y_train, y_pred=X_pred))
print('accuracy = ', accuracy_score(y_true=Y_train, y_pred=X_pred))
print('precision = ', precision_score(y_true=Y_train, y_pred=X_pred,average='macro'))
print('recall = ', recall_score(y_true=Y_train, y_pred=X_pred,average='macro'))
print('f1 score = ', f1_score(y_true=Y_train, y_pred=X_pred,average='macro'))

print()

print('test:')
print('confusion matrix = \n', confusion_matrix(y_true=Y_test, y_pred=Y_pred))
print('accuracy = ', accuracy_score(y_true=Y_test, y_pred=Y_pred))
print('precision = ', precision_score(y_true=Y_test, y_pred=Y_pred,average='macro'))
print('recall = ', recall_score(y_true=Y_test, y_pred=Y_pred,average='macro'))
print('f1 score = ', f1_score(y_true=Y_test, y_pred=Y_pred,average='macro'))

print() """

# Twitter analysis
new=pd.read_csv('data/Lesson_analysis.tsv',sep='\t')
new=new[new.sentiment!=0]
text=new.tweet

text=vectorizer.transform(text)
text=pd.DataFrame(text.toarray(),columns=vocab).to_numpy()
print(text.shape)

Twitter_pred=lr.predict(text)

Twitter_right=new.sentiment

for index,text in enumerate(Twitter_pred):
    if(Twitter_right[index]!=Twitter_pred[index])
        print(new.tweet[index])
        print('right='+Twitter_right[index],'wrong='+Twitter_pred[index])

print('Twitter:')
print('confusion matrix = \n', confusion_matrix(y_true=Twitter_right, y_pred=Twitter_pred))
print('accuracy = ', accuracy_score(y_true=Twitter_right, y_pred=Twitter_pred))
print('precision = ', precision_score(y_true=Twitter_right, y_pred=Twitter_pred,average='binary'))
print('recall = ', recall_score(y_true=Twitter_right, y_pred=Twitter_pred,average='binary'))
print('f1 score = ', f1_score(y_true=Twitter_right, y_pred=Twitter_pred,average='binary'))







#plotする
# X_train_plot=np.vstack(X_train)
# train_label_plot=np.hstack(train_label)
# X_test_plot=np.vstack(X_test)
# test_label_plot=np.hstack(test_label)
#plot_decision_regions(X_train_plot, train_label_plot, clf=clf_result, res=0.01) #学習データをプロット
# plot_decision_regions(X_test_plot, test_label_plot, clf=clf_result, res=0.01, legend=2) #テストデータをプロット

# 8：任意のデータに対する識別結果を見てみる------------------
#predicted_label=clf_result.predict([1,-1])
#print("このテストデータのラベル = ", predicted_label)
 
# 9：識別平面の式を手に入れる--------------------------------
# print(clf_result.intercept_)
# print(clf_result.coef_ )  #coef[0]*x+coef[1]*y+intercept=0