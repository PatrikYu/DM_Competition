#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
#
# %matplotlib inline   # 两种方法都报错

"""
数据分析，特征工程，建模分别写作一个py文件，查看输出或者一步一步处理的时候可以在Python Console里面写
"""
# 一、数据分析


# 导入相关数据包
import numpy as np
import pandas as pd
import seaborn as sns # Seaborn是基于matplotlib的Python可视化库
import matplotlib.pyplot as plt

train = pd.read_csv('F:/competition/titanic/train.csv')
test = pd.read_csv('F:/competition/titanic/test.csv')

# 特征详情

# # 查看头五行
# print train.head(5)
# # 返回每列列名,该列非空值个数,以及该列类型
# print train.info()   # 可见，有不少样本的特征值存在缺失
# 返回特征值（数值型变量）的统计量
# train.describe(percentiles=[0.00, 0.25, 0.5, 0.75, 1.00])  # 分位数：如中位数，数值大小为中间值的那个数
# print train.describe() # 返回非缺失值的数量，均值，标准差，以及前%0（最小值）,0.25,0.5（中位数）...等的值

# 特征分析（统计学与绘图）

# 查看存活人数
print train['Survived'].value_counts() # 这一特征值为0或1，统计1的个数，即存活人数
# 相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.
train_corr = train.drop('PassengerId',axis=1).corr() # drop函数默认删除行，删除列加上 axis=1
# 画出相关性热力图
a = plt.subplots(figsize=(15,9))#调整画布大小
a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)#画热力图
# 进一步探索分析各个数据与结果的关系
# 乘客等级与存活率的关系，Pclass,乘客等级,1是最高级
train.groupby(['Pclass'])['Pclass','Survived'].mean()
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()
# 可以看出Survived和Pclass在Pclass=1的时候有较强的相关性（>0.5），所以最终模型中包含该特征
# Sex,性别
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
# Age年龄与生存情况的分析
g = sns.FacetGrid(train, col='Survived',size=5) # 还可以添加hue参数：每个横坐标上画的bar个数=hue上属性的取值个数
# FacetGrid是seaborn中一个绘制多个图表（以网格形式显示）的接口，图表个数=survived的取值个数
g.map(plt.hist, 'Age', bins=40) # 横坐标为age，map：将绘图功能应用于每个方面的数据子集。
train.groupby(['Age'])['Survived'].mean().plot()
sns.coutplot('Embarked',hue='Survived',data=train) # 横坐标为港口名称，在每个港口上做多个bar（个数=Survived的取值个数2）
# plt.show()


# 二、特征工程


#先将数据集合并,一起做特征工程(注意,标准化的时候需要分开处理)
#先将test补齐,然后通过pd.apped()合并
test['Survived'] = 0
train_test = train.append(test)

train_test = pd.get_dummies(train_test,columns=['Pclass']) # 分列处理
train_test = pd.get_dummies(train_test,columns=['Sex'])
# print train_test.head(5)    # 实现分栏处理，去掉原来的Sex那一列，在末尾加上两列分别为 Sex_female 和 Sex_male
train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']  # 创造新特征：在船上认识的人数
train_test = pd.get_dummies(train_test,columns = ['SibSp','Parch','SibSp_Parch'])
train_test = pd.get_dummies(train_test,columns=["Embarked"])

# 1.在数据的Name项中包含了对该乘客的称呼,将这些关键词提取出来,然后做分列处理.
#从名字中提取出称呼： df['Name'].str.extract()是提取函数,配合正则一起使用，str.strip()去除首尾空格
train_test['Name1'] = train_test['Name'].str.extract('.+,(.+)', expand=False).str.extract('^(.+?)\.', expand=False).str.strip()
# 将 , 之后 . 之前的称谓提取出来，如 Mr Mrs
# print train_test['Name1'].head(5)
#将姓名分类处理()
train_test['Name1'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer' , inplace = True)
train_test['Name1'].replace(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty' , inplace = True)
train_test['Name1'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs') # 注意相似的称谓
train_test['Name1'].replace(['Mlle', 'Miss'], 'Miss')
train_test['Name1'].replace(['Mr'], 'Mr' , inplace = True)
train_test['Name1'].replace(['Master'], 'Master' , inplace = True)
#分列处理
train_test = pd.get_dummies(train_test,columns=['Name1'])
# print train_test.head(5)

# 从姓名中提取出姓做特征（目的是为了提取同一姓氏的人数，姓氏人数可能跟你的人脉，聚集地有关，也会影响你的逃生顺序）
#从姓名中提取出姓，apply+lambda函数
train_test['Name2'] = train_test['Name'].apply(lambda x: x.split('.')[1]) # 取第二个，索引为1的，才是姓氏

#计算数量,然后合并数据集
Name2_sum = train_test['Name2'].value_counts().reset_index()
# 统计各种姓氏的数量，并重置索引（将Name2的所有数值列为单独一列，然后后面跟着他们的数量）
# print Name2_sum.head(5)
Name2_sum.columns=['Name2','Name2_sum']   # 将姓氏记为Name2列 ，之前的姓氏计数值记为 Name2_sum列
# print Name2_sum.head(5)
train_test = pd.merge(train_test,Name2_sum,how='left',on='Name2')
# how="left"就是说，保留Name2字段的全部信息，不增加也不减少，但是拼接的时候只把df2表中的与df1中Name2字段交集的部分合并上就可以了
# 这样相当于将数量那一项加上去了
# 得到的train_test是pandas中的一种数据结构，叫做 DataFrame ，numpy中的array或者ndarray并没有 head(5)的操作方法
# print train_test.head(5)
#由于出现一次时该特征是无效特征,用one来代替出现一次的姓
# loc函数主要通过行标签索引行数据，将选中的索引的 Name2_new 列用 'one'来代替
train_test.loc[train_test['Name2_sum'] == 1 , 'Name2_new'] = 'one'
train_test.loc[train_test['Name2_sum'] > 1 , 'Name2_new'] = train_test['Name2']
del train_test['Name2']
#分列处理
train_test = pd.get_dummies(train_test,columns=['Name2_new'])
#删掉姓名这个特征
del train_test['Name']
print train_test.head(5)

# fare（票价）该特征有缺失值,先找出缺失值的那调数据,然后用平均数填充
#从上面的分析,发现该特征train集无miss值,test有一个缺失值,先查看
print train_test.loc[train_test["Fare"].isnull()]     # 可见 序号为 1043 的项的船票缺失，这应该是测试样本里面的

#票价与pclass和Embarked有关,所以用train分组后的平均数填充
print train.groupby(by=["Pclass","Embarked"]).Fare.mean()   # 得到三个等级对应的三个Embarked（登船港口）票价平均值，共9行
#用pclass=3和Embarked=S的平均数14.644083来填充
train_test["Fare"].fillna(14.435422,inplace=True)  # 序号为 1043 的样本 为3等船票+S港上船，会自动填充到null值的位置

#将Ticket提取字符列
#str.isnumeric()  如果S中只有数字字符(则不符合标准，用np中的nan代替)，则返回True，否则返回False
# 注意isnumeric()仅仅适用于unicode，而python的str默认是ascii编码，所以可以用unicode()将ascii转换为unicode
# 或者直接使用ascii适用的 isdigit() 函数
train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isdigit() else x)
train_test.drop('Ticket',inplace=True,axis=1)
#分列,此时nan值可以不做处理
train_test = pd.get_dummies(train_test,columns=['Ticket_Letter'],drop_first=True) # 为什么要drop_first？？？实在没搞懂

# Age这一栏有太多缺失值，考虑用一个回归模型进行填充.
# 在模型修改的时候,考虑到年龄缺失值可能影响死亡情况（可能是下层人士，没有登记年龄）,用年龄是否缺失值来构造新特征
#考虑年龄缺失值可能影响死亡情况,数据表明,年龄缺失的死亡率为0.19."""
print train_test.loc[train_test["Age"].isnull()]['Survived'].mean()
# 所以用年龄是否缺失值来构造新特征
train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1    # 缺失构造特征值为1
train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0   # 未缺失构造为0
train_test = pd.get_dummies(train_test,columns=['age_nan'])
# 利用其他组特征量，采用机器学习算法来预测Age
train_test.info()     # 看看现在数据集的情况
#创建没有['Survived','Cabin']的数据集
missing_age = train_test.drop(['Survived','Cabin'],axis=1)
#将Age完整的项作为训练集、将Age缺失的项作为测试集。
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]
#构建训练集合预测集的X和Y值
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
missing_age_Y_train = missing_age_train['Age']
missing_age_X_test = missing_age_test.drop(['Age'], axis=1)
# 先将数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler() # 默认参数
ss.fit(missing_age_X_train)  # 取missing_age_X_train的均值或方差作为标准化的依据
missing_age_X_train = ss.transform(missing_age_X_train) # 对训练集做标准化
missing_age_X_test = ss.transform(missing_age_X_test) # 对测试集做标准化
#使用贝叶斯预测年龄
from sklearn import linear_model
lin = linear_model.BayesianRidge() # 默认参数
lin.fit(missing_age_X_train,missing_age_Y_train)  # 训练模型
# 调参有必要学习一下！！！
# BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
#         fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
#         normalize=False, tol=0.001, verbose=False)
#利用loc将预测值填入数据集
train_test.loc[(train_test['Age'].isnull()), 'Age'] = lin.predict(missing_age_X_test)
#用pd.cut函数将年龄划分为5个阶段：10以下,10-18,18-30,30-50,50以上
train_test['Age'] = pd.cut(train_test['Age'], bins=[0,10,18,30,50,100],labels=[1,2,3,4,5])
train_test = pd.get_dummies(train_test,columns=['Age'])  # 分列处理

# cabin：客舱号码
#cabin项缺失太多，第一种：按Cain首字母进行分类,缺失值为一类,作为特征值进行建模
train_test['Cabin_nan'] = train_test['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else x)
train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
#第二种：将有无Cain进行分类,
train_test.loc[train_test["Cabin"].isnull() ,"Cabin_nan"] = 1   # 缺失cabin为1
train_test.loc[train_test["Cabin"].notnull() ,"Cabin_nan"] = 0
train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
# 也可以考虑直接舍去该特征
train_test.drop('Cabin',axis=1,inplace=True) # 丢掉此特征

#  特征工程处理完了,划分数据集
train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)

# 数据规约 ：又称为数据正则化，将样本的某个范数缩放到单位1，这是针对单个样本的，对每个样本将它缩放到单位范数
# 可以用来得到数据集的规约表示，使得数据集变小，但同时仍然近于保持原数据的完整性，常用方法：
# 1.线性模型需要用标准化的数据建模,而树类模型不需要标准化的数据
# 2.处理标准化的时候,注意将测试集的数据transform到test集上
# 如：进行L1范数规约，就对此样本上的每个特征都除以他们特征值之和
from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
ss2.fit(train_data_X)
train_data_X_sd = ss2.transform(train_data_X)
test_data_X_sd = ss2.transform(test_data_X)
# 此时得到的 test_data_X_sd 数据类型为 numpy.ndarray




# 三. 构建模型



# 模型发现
# 可选单个模型模型有随机森林,逻辑回归,svm,xgboost,gbdt等.
# 也可以将多个模型组合起来,进行模型融合,比如voting,stacking等方法
# 好的特征决定模型上限,好的模型和参数可以无线逼近上限.
# 我测试了多种模型,模型结果最高的随机森林,最高有0.8.

# 随机森林

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6,oob_score=True)
# obb_score：是否使用袋外样品来估计泛化精度。
rf.fit(train_data_X,train_data_Y) # 训练

test["Survived"] = rf.predict(test_data_X)
RF = test[['PassengerId','Survived']].set_index('PassengerId')
# 以'PassengerID'的所有取值作为索引
RF.to_csv('RF.csv')
# 随机森林是随机选取特征进行建模的,所以每次的结果可能都有点小差异
# 如果分数足够好,可以将该模型保存起来,下次直接调出来使用0.81339 'rf10.pkl'
from sklearn.externals import joblib
joblib.dump(rf, 'rf10.pkl')
# clf = joblib.load('rf10.pkl')        # 加载保存的模型
# test["Survived"] = clf.predict(test_data_X)


# 四.建立模型



# 模型融合 voting 投票：


from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.1,max_iter=100)

import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=6,min_samples_leaf=2,n_estimators=100,num_round = 5)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)

vot = VotingClassifier(estimators=[('lr', lr), ('rf', rf),('gbdt',gbdt),('xgb',xgb_model)], voting='hard')
vot.fit(train_data_X_sd,train_data_Y)

test["Survived"] = vot.predict(test_data_X_sd)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('vot5.csv')



# 模型融合 stacking


# 划分train数据集,调用代码,把数据集名字转成和代码一样
X = train_data_X_sd
X_predict = test_data_X_sd
y = train_data_Y

#    '''模型融合中使用到的各个单模型'''
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

clfs = [LogisticRegression(C=0.1,max_iter=100),
        xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]

# 创建n_folds
from sklearn.cross_validation import StratifiedKFold
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))

# 创建零矩阵
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

# 建立模型
for j, clf in enumerate(clfs): # j是模型编号，clfs是包含各个模型的list
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf): # i是训练样本被分割后的第i部分
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        """模型j对训练样本的预测存储在j列中，行数：训练样本个数，列数：模型个数"""
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''模型j对测试样本的预测存储在j列中。对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

# 建立第二层模型（此处采用逻辑回归）
clf2 = LogisticRegression(C=0.1,max_iter=100) # C=正则化系数的倒数
clf2.fit(dataset_blend_train, y) # y = train_data_Y
y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]

test = pd.read_csv("test.csv")
test["Survived"] = clf2.predict(dataset_blend_test)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('stack3.csv')
