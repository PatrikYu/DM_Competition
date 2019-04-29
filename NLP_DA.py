#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

"""
https://github.com/apachecn/kaggle/tree/dev/competitions/getting-started/word2vec-nlp-tutorial
"""

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

train = pd.read_csv('F:/competition/NLP/labeledTrainData.tsv',header=0, delimiter="\t", quoting=3)
test = pd.read_csv('F:/competition/NLP/testData.tsv',header=0, delimiter="\t", quoting=3)
# csv文件为用,分隔的文件, tsv为用制表符分隔的文件
# header指定行数用来作为列名，header=0表示数据的第一行是属性值，delimiter分隔符, 指定文件里面的元素是用什么分隔的
# quoting控制csv中的引号常量

print(train.shape)
print(train.columns.values)
print(train.head(3))

# 数据清洗和文本预处理

# 去除评论中的HTML标签：BeautifulSoup包
print('\n处理前: \n', train['review'][0])
example1 = BeautifulSoup(train['review'][0], "html.parser")
# 处理标点符号, 数字 : 正则表达式
# NLTK处理停用词(stopword)：那些出现频率高,但是却没有多大意义的单词
import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub('[^a-zA-Z]',  # 搜寻的pattern
                      ' ',           # 用来替代的pattern(空格)
                      example1.get_text())  # 待搜索的text
print(letters_only)
lower_case = letters_only.lower()  # Convert to lower case 所有大写字母转为小写
words = lower_case.split()  # Split into word

print('\n处理后: \n', words)

def review_to_wordlist(review):
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    # 用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 小写化所有的词，并转成词list
    words = review_text.lower().split()
    # 返回words
    return words

# 预处理数据
label = train['sentiment']
train_data = []
for i in range(len(train['review'])):
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))
    # 以空格来连接每一个分隔开的单词并放在 'review'栏中
test_data = []
for i in range(len(test['review'])):
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))

# 预览数据
print(train_data[0], '\n')
print(test_data[0])


# 特征处理
# 把文本转换为向量

# TF-IDF向量
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
# 参考：http://blog.csdn.net/longxinchen_ml/article/details/50629613

"""
min_df: 最小支持度为2（词汇出现的最小次数）
max_features: 默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集
strip_accents: 将使用ascii或unicode编码在预处理步骤去除raw document中的重音符号
analyzer: 设置返回类型
token_pattern: 表示token的正则表达式，需要设置analyzer == 'word'，默认的正则表达式选择2个及以上的字母或数字作为token，
               标点符号默认当作token分隔符，而不会被当作token
ngram_range: 词组切分的长度范围
use_idf: 启用逆文档频率重新加权
use_idf：默认为True，权值是tf*idf，如果设为False，将不使用idf，就是只使用tf，相当于CountVectorizer了，词频：词条在文档d中出现的频率
         IDF逆向文件频率是一个词语普遍重要性的度量。某一特定词语的IDF，可以由总文件数目除以包含该词语之文件的数目，
         再将得到的商取以10为底的对数得到
smooth_idf: idf平滑参数，默认为True，idf=ln((文档总数+1)/(包含该词的文档数+1))+1，如果设为False，idf=ln(文档总数/包含该词的文档数)+1
sublinear_tf: 默认为False，如果设为True，则替换tf为1 + log(tf)
stop_words: 设置停用词，设为english将使用内置的英语停用词，设为一个list可自定义停用词，设为None不使用停用词，
            设为None且max_df∈[0.7, 1.0)将自动根据当前的语料库建立停用词表
"""
tfidf = TFIDF(min_df=2,
           max_features=None,
           strip_accents='unicode',
           analyzer='word',
           token_pattern=r'\w{1,}', # 一个单词或数字以上的作为token
           ngram_range=(1, 3),  # 二元文法模型
           use_idf=1,
           smooth_idf=1,
           sublinear_tf=1,
           stop_words = 'english') # 去掉英文停用词

# 合并训练和测试集以便进行TFIDF向量化操作
data_all = train_data + test_data
len_train = len(train_data)

tfidf.fit(data_all)
data_all = tfidf.transform(data_all) # 得到tf-idf矩阵，稀疏矩阵表示法
# print (data_all.todense()) # 转换为更直观的一般矩阵
# 总的列数就是所有文档中词汇种类的个数，每一行对应一个文档，每一列对应着某个单词的tf-idf值
# 恢复成训练集和测试集部分
train_x = data_all[:len_train]
test_x = data_all[len_train:]
print('TF-IDF处理结束.')

print("train: \n", np.shape(train_x[0]))
print("test: \n", np.shape(test_x[0]))

"""
在 Python Console 中进行下一步操作
import DA.py # 执行一遍，也可以直接用第二句
from DA import train_x,label
在这个基础上用各类分类器去训练，不能每次都跑一遍程序，这样太慢了

"""
# 朴素贝叶斯训练

from sklearn.naive_bayes import MultinomialNB as MNB

model_NB = MNB() # (alpha=1.0, class_prior=None, fit_prior=True)
# 为了在预测的时候使用
model_NB.fit(train_x, label)

from sklearn.model_selection import cross_val_score
import numpy as np

print(u"多项式贝叶斯分类器10折交叉验证得分:  \n", cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc'))
print(u"\n多项式贝叶斯分类器10折交叉验证平均得分: ", np.mean(cross_val_score(model_NB, train_x, label, cv=10, scoring='roc_auc')))
# 加个u就转换为unicode，可以正常输出中文了

from DA import test,test_x
test_predicted = np.array(model_NB.predict(test_x))
print(u'保存结果...')
# 生成预测结果表格
import pandas as pd
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))
submission_df.to_csv('F:/competition/NLP/submission_br.csv',columns = ['id','sentiment'], index = False)

# 逻辑回归

from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV
# 超参数自动搜索模块，跑ML时通常待调节的参数有很多，参数之间的组合更是繁复
# GridSearchCV模块，能够在指定的范围内自动搜索具有不同超参数的不同模型组合
# 设定grid search的参数
grid_values = {'C': [1, 15, 30, 50]}
# grid_values = {'C': [30]}
# 'C'是正则化强度值，与SVM一样，较小的值指定更强的正则化，默认为1
"""
penalty: l1 or l2, 用于指定惩罚中使用的标准。对应的是绝对值惩罚还是平方项惩罚

"""
model_LR = GridSearchCV(LR(penalty='l2', dual=True, random_state=0), grid_values, scoring='roc_auc', cv=20)
model_LR.fit(train_x, label)
# 20折交叉验证
# GridSearchCV(cv=20,
#         estimator=LR(C=1.0,
#             class_weight=None,
#             dual=True,
#             fit_intercept=True,
#             intercept_scaling=1,
#             penalty='l2',
#             random_state=0,
#             tol=0.0001),
#         fit_params={},
#         iid=True,
#         n_jobs=1,
#         param_grid={'C': [30]},
#         pre_dispatch='2*n_jobs',
#         refit=True,
#         scoring='roc_auc',
#         verbose=0)
print(model_LR.cv_results_, '\n', model_LR.best_params_, model_LR.best_score_)
"""显示出四个C对应的20次平均拟合和打分时间，以及20次每次得到的四个参数对应的准确率
    最后输出最好的参数以及对应的测试样本上20次实验所得的平均准确率 """

test_predicted = np.array(model_LR.predict(test_x))
print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': test_predicted})
print(submission_df.head(10))

# 结果显示：逻辑回归的结果相对贝叶斯模型有所提高


# word2vec向量


# 神经网络语言模型L = SUM[log(p(w|contect(w))]，即在w的上下文下计算当前词w的概率，由公式可以看到，
# 我们的核心是计算p(w|contect(w)， Word2vec给出了构造这个概率的一个方法。

import nltk
from nltk.corpus import stopwords

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# tokenizer是个分句器，这里导入了英文的分句器，注意：每一句话结束的标点后老外都会留一个空格，没留空格的，分词器认为这是一句话
# 貌似nltk.data中还没有中文的分句器

def review_to_wordlist(review, remove_stopwords=False):
    # review = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review)
    words = review_text.lower().split()
    if remove_stopwords: # 移去停止词
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # print(words)
    return (words)


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    '''
    1. 将评论文章，按照句子段落来切分(所以会比文章的数量多很多)
    2. 返回句子列表，每个句子由一堆词组成
    '''
    review = BeautifulSoup(review, "html.parser").get_text()
    # raw_sentences 句子段落集合
    raw_sentences = tokenizer.tokenize(review) # 切分段落为句子
    # print(raw_sentences)
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            # 对于每一个句子来切分，获取句子中的词列表
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

sentences = []
for i, review in enumerate(train["review"]):
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # print(i, review) 评论可能是几段话
    sentences += review_to_sentences(review, tokenizer, True)
print(np.shape(train["review"]))
print(np.shape(sentences))

unlabeled_train = pd.read_csv("%s/%s" % ("F:/competition/NLP", "unlabeledTrainData.tsv"), header=0, delimiter="\t", quoting=3 )
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)
print('预处理 unlabeled_train data...')
print(np.shape(train["review"]))
print(np.shape(sentences))

# 构建word2vec模型

import time
from gensim.models import Word2Vec
# word2vec词向量工具是由google开发的，输入为文本文档，输出为基于这个文本文档的语料库训练得到的词向量模型
# 可以很好地对词之间的相似性进行度量。每个词都对应着一个向量（而不是常规的稀疏向量），包含着词与词之间关系的信息
# 模型参数
num_features = 300    # Word vector dimensionality 特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好
min_word_count = 40   # Minimum word count 词频少于min_count次数的单词会被丢弃掉, 默认值为5
num_workers = 4       # Number of threads to run in parallel 控制训练的并行数
context = 10          # Context window size 当前词与预测词在一个句子中的最大距离是多少
downsampling = 1e-3   # Downsample setting for frequent words 高频词汇的随机降采样的配置阈值，默认为1e-3，范围是(0,1e-5)

%%time
# 训练模型（得到每个词的词向量表示）
print("训练模型中...")
model = Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count=min_word_count, \
            window=context, sample=downsampling)
print("训练完成")

print('保存模型...')
model.init_sims(replace=True)
model_name = "%s/%s" % ("F:/competition/NLP", "300features_40minwords_10context")
model.save(model_name)
print('保存结束')

# 预览模型

model.doesnt_match("man woman child kitchen".split()) # 取出不匹配的那一项
model.doesnt_match("france england germany berlin".split())
model.most_similar("man") # 取出与"man"相似度最高的那一项
model.wv.most_similar("awful", topn=5)
model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
# 与 'woman', 'king' 相似，与'man'不相似的1项

# 使用Word2vec特征

def makeFeatureVec(words, model, num_features):
    '''
    对段落中的所有词向量进行取平均操作
    '''
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word包含了词表中的所有词，为了检索速度，保存到set中
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # 取平均
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    '''
    给定一个文本列表，每个文本由一个词列表组成，返回每个文本的词向量平均值
    '''
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")

    for review in reviews:
        if counter % 5000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))

        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1

    return reviewFeatureVecs # 得到这段评论对应的向量

%time trainDataVecs = getAvgFeatureVecs(train_data, model, num_features)
print(np.shape(trainDataVecs))
%time testDataVecs = getAvgFeatureVecs(test_data, model, num_features)
print(np.shape(testDataVecs))
# 得到一个 (25000, 300) 的矩阵，一共有25000个评论，每个评论用一个长度为300的向量来表示

# 高斯贝叶斯+Word2vec训练
from sklearn.naive_bayes import GaussianNB as GNB

model_GNB = GNB()
model_GNB.fit(trainDataVecs, label)

from sklearn.cross_validation import cross_val_score
import numpy as np

print("高斯贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model_GNB, trainDataVecs, label, cv=10, scoring='roc_auc')))

print('保存结果...')
result = model_GNB.predict( testDataVecs )
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': result})
print(submission_df.head(10))
submission_df.to_csv('/Users/jiangzl/Desktop/gnb_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

"""
从验证结果来看，没有超过基于TF-IDF多项式贝叶斯模型
"""

# 随机森林+Word2vec训练

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier( n_estimators = 100, n_jobs=2)

print("Fitting a random forest to labeled training data...")
%time forest = forest.fit( trainDataVecs, label )
print("随机森林分类器10折交叉验证得分: ", np.mean(cross_val_score(forest, trainDataVecs, label, cv=10, scoring='roc_auc')))

# 测试集
result = forest.predict( testDataVecs )

print('保存结果...')
submission_df = pd.DataFrame(data ={'id': test['id'], 'sentiment': result})
print(submission_df.head(10))
submission_df.to_csv('/Users/jiangzl/Desktop/rf_word2vec.csv',columns = ['id','sentiment'], index = False)
print('结束.')

"""
改用随机森林之后，效果有提升，但是依然没有超过基于TF-IDF多项式贝叶斯模型
"""


"""
利用 TSNE 和 matplotlib 对分类结果进行可视化处理...

利用 matplotlib 和 metric 库来构建 ROC 曲线...

"""

