# coding=utf-8

import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import preprocessing.preprocesser as pp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
#from sklearn.feature_selection import SelectKBest, SelectPercentile
#from sklearn.feature_selection import chi2,f_classif,mutual_info_classif
# from sklearn.feature_extraction import DictVectorizer # 定义一组字典列表，用来表示多个数据样本（每个字典代表一个数据样本）
# sklearn.metric 里有距离计算
# sklearn.preprocessing 里有scale方法 import StandardScaler sc = StandardScaler() X_train = sc.fit_transform(X_train)
# from sklearn.pipeline import Pipeline clf = Pipeline([('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),('classification', RandomForestClassifier())]) clf.fit(X, y)

class ClPreprocessor(pp.Preprocessor):

    def __init__(self, path, file_suffix):
        pp.Preprocessor.__init__(self, path, file_suffix) # 继承父类， 也可以写为 super(ClPreprocessor,self).__init__(path,format)
        my_stop_words = text.ENGLISH_STOP_WORDS
        my_stop_words.union(['的','我','了','啊','quot','&'])
        self.count_vectorizer = CountVectorizer(stop_words=my_stop_words, ngram_range=(1,1))
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=my_stop_words, ngram_range=(1,1))

    def split_file(self, file_name, file_suffix):
        raw_data_pd = pd.read_csv(self.path+file_name, sep='###',header=0)
        # 采样均衡数据
        corpus_pos = raw_data_pd[raw_data_pd.label == 1]
        corpus_neg = raw_data_pd[raw_data_pd.label == 0]
        sample_size = min(corpus_pos.shape[0], corpus_neg.shape[0])
        raw_data_pd = self.get_balance_corpus(sample_size, corpus_pos, corpus_neg)
        # 分割数据集
        x, y = raw_data_pd['review'],raw_data_pd['label']
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        train_data = pd.concat([x_train,y_train], axis=1)
        test_data = pd.concat([x_test,y_test], axis=1)
        # 写入文件
        train_data.to_csv(self.path+'train.txt', sep='$', index=False)
        test_data.to_csv(self.path+'test.txt', sep='$', index=False)

    def get_balance_corpus(self, sample_size, corpus_pos, corpus_neg):
        # replace = True 有放回采样
        pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size), \
                                       corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)])

        print('（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
        print('（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])

        return pd_corpus_balance

    def get_train_data(self):
        train_data_pd = pd.read_csv(self.path + 'train.txt', sep='$',header=0)
        x, y = train_data_pd['review'], train_data_pd['label']
        x_feature = self.get_text_feature(x)
        return x_feature, y

    def get_test_data(self):
        test_data_pd = pd.read_csv(self.path + 'test.txt', sep='$',header=0)
        x, y = test_data_pd['review'], test_data_pd['label']
        x_feature = self.get_text_feature(x, False)
        return x_feature, y

    def get_text_feature(self, x, is_train=True):
        x_cut = [' '.join(list(jieba.cut(w, cut_all=False))) for w in x]
        #x_feature = self.count_vectorizer.fit_transform(x_cut).todense()  # todense将稀疏矩阵转化为完整特征矩阵
        if is_train:
            x_feature = self.tfidf_vectorizer.fit_transform(x_cut).todense()
        else:
            x_feature = self.tfidf_vectorizer.transform(x_cut).todense()
        # idf_values = dict(zip(self.tfidf_vectorizer.get_feature_names(), self.tfidf_vectorizer.idf_))
        print(x_feature)
        return x_feature
