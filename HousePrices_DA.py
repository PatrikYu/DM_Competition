#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

"""
官方教程：https://github.com/apachecn/kaggle/tree/dev/competitions/getting-started/house-price
DA.py是根据官方教程写的，实际上有些步骤被省略了，数据分析部分参考
https://blog.csdn.net/Irving_zhang/article/details/78561105

"""


# 一、数据分析




# 导入相关数据包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

# 读入数据
train = pd.read_csv('F:/competition/house prices/train.csv')
test = pd.read_csv('F:/competition/house prices/test.csv')

# 特征说明
train.columns
# train.columns # 显示各个特征值的非空值数量情况

# 特征详情
# print train.head(5)

# 特征分析（统计学与绘图）
# 相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.
train_corr = train.drop('Id',axis=1).corr() # 列（即特征值）的相关性
# print train_corr
a = plt.subplots(figsize=(20, 12))#调整画布大小
a = sns.heatmap(train_corr, vmax=.8, square=True)#画热力图   annot=True 显示系数

# 寻找K个最相关的特征信息（相关度很高的两个特征取其中一个即可）
k = 10 # number of variables for heatmap
cols = train_corr.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.5)
hm = plt.subplots(figsize=(20, 12))#调整画布大小
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values)
# annot_kws={'size'热力图方框中字体的大小
# plt.show()

# SalePrice和相关变量之间的散点图
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)  # 还能使用回归等,kind="reg"
# plt.show()
train[['SalePrice', 'OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']].info()
# P11柱形图依照价格自动分为若干段，高度代表样本数量。P12散点图代表以overallQual为横坐标，价格为纵坐标的点样本分布情况



# 二、特征工程




#先将数据集合并,一起做特征工程(注意,标准化的时候需要分开处理)
#先将test补齐,然后通过pd.apped()合并
test['SalePrice'] = None
train_test = pd.concat((train, test)).reset_index(drop=True)
# 简单拼接表单，默认axis=0，按行拼接；reset_index重新变为默认的整型索引，drop=true，代表丢弃索引列

# 缺失值分析
total= train_test.isnull().sum().sort_values(ascending=False)
# isnull生成所有数据的true/false矩阵，ascending=False降序,
# 注意：Series或DataFrame的sum函数默认按照每列（即每个特征值）来统计数量，count函数将每轴的数量加起来。
percent = (train_test.isnull().sum()/train_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total','Lost Percent'])
# 按列拼接，列的名称分别为 'Total','Lost Percent'

print(missing_data[missing_data.isnull().values==False].sort_values('Total', axis=0, ascending=False).head(20))
# 仅输出isnull函数中值为false（即个数非空）的项，并降序排列输出前二十项
# 1. 对于缺失率过高的特征，例如 超过15% 我们应该删掉相关变量且假设该变量并不存在
# 2. GarageX 变量群的缺失数据量和概率都相同，可以选择一个就行，例如：GarageCars
# 3. 对于缺失数据在5%左右（缺失率低），可以直接删除/回归预测

train_test = train_test.drop((missing_data[missing_data['Lost Percent'] > 0.15]).index.drop('SalePrice') , axis=1)
# 将遗失数据（百分比在0.15以上）SalePrice这列上的数据清除
# train_test = train_test.drop(train.loc[train['Electrical'].isnull()].index)

tmp = train_test[train_test['SalePrice'].isnull().values==False]
# 保留原始样本集中SalePrice不为0的项
print(tmp.isnull().sum().max()) # just checking that there's no missing data missing

# 异常值处理

# 单因素分析
# 首先对数据进行正态化，意味着将数据值转换为均值为0，方差为1的数据

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(train.SalePrice) # 原始数据分布直方图,['SalePrice']和.SalePrice效果是一样的
ax2.hist(np.log1p(train.SalePrice)) # 正态化之后分布
# plt.show()
# 数据偏度和峰度度量：
print("Skewness: %f" % train['SalePrice'].skew()) # 偏度
print("Kurtosis: %f" % train['SalePrice'].kurt()) # 峰度

# 双变量分析

# 1.GrLivArea 和 SalePrice 双变量分析
var = 'GrLivArea'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# 删除点
print(train.sort_values(by='GrLivArea', ascending = False)[:2])
# 增序排列，输出头两个GrLivArea值特别高的（离群点），找到他俩的id，接下来删除
train_test = train_test.drop(tmp[tmp['Id'] == 1299].index)
train_test = train_test.drop(tmp[tmp['Id'] == 524].index)

# 2.TotalBsmtSF 和 SalePrice 双变量分析

var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'],train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice',ylim=(0,800000))

# 房价SalePrice如何遵循统计假设，测量它的四个假设量

# 正态性
# 1、SalePrice 绘制直方图和正态概率图：
sns.distplot(train['SalePrice'],fit=st.norm)
# distplot绘制直方图，, fit控制拟合的参数分布图形(此处做了归一化)
fig = plt.figure()
res = st.probplot(train['SalePrice'], plot=plt)
# 进行对数变换（loglp代表正态化：均值为0，方差为1）：
train_test['SalePrice'] = [i if i is None else np.log1p(i) for i in train_test['SalePrice']]
# train_test['TotalBsmtSF']= np.log1p(train_test['TotalBsmtSF']) 另一种更简便的语法
# 绘制变换后的直方图和正态概率图：
# 注：正态概率图用于检查一组数据是否服从正态分布，是实数与正态分布数据之间函数关系的散点图。
#     若符合正态分布，正态概率图将是一条直线。
tmp = train_test[train_test['SalePrice'].isnull().values==False]

sns.distplot(tmp[tmp['SalePrice'] !=0]['SalePrice'],fit=st.norm);
fig = plt.figure()
res = st.probplot(tmp['SalePrice'], plot=plt)
# plt.show()

# 2、GrLivArea （绘图完全与1相同）
# 3、TotalBsmtSF
sns.distplot(train['TotalBsmtSF'],fit=st.norm);
fig = plt.figure()
res = st.probplot(train['TotalBsmtSF'],plot=plt)
# 大量为 0(Y值) 的观察值（没有地下室的房屋）,含 0(Y值) 的数据无法进行对数变换
# 去掉为0的分布情况
# 这里的索引函数loc[某一行，某一列]
tmp1 = np.array(tmp.loc[tmp['TotalBsmtSF']>0, ['TotalBsmtSF']]) # 这里得到的是一个列向量，[1]/n[23]/n...
tmp = np.array(tmp.loc[tmp['TotalBsmtSF']>0, ['TotalBsmtSF']])[:, 0] # [:,0]就将它转换为了行向量 [1,23,4,5...]

# 对'TotalBsmtSF'这一栏取大于0的项
sns.distplot(tmp, fit=st.norm)
fig = plt.figure()
res = st.probplot(tmp, plot=plt)

# 我们建立了一个变量，可以得到有没有地下室的影响值（二值变量），我们选择忽略零值，只对非零值进行对数变换。
# 这样我们既可以变换数据，也不会损失有没有地下室的影响。
# print(train.loc[train['TotalBsmtSF']==0, ['TotalBsmtSF']])  # 输出为0的项
print(train.loc[train['TotalBsmtSF']==0, ['TotalBsmtSF']].count()) # 输出TotalBsmtSF为0的项的个数
train.loc[train['TotalBsmtSF']==0,'TotalBsmtSF'] = 1 # 改为1，这样取对数以后为0
print(train.loc[train['TotalBsmtSF']==1, ['TotalBsmtSF']].count())

# 进行对数变换：
train_test['TotalBsmtSF']= np.log1p(train_test['TotalBsmtSF'])
tmp = train_test[train_test['SalePrice'].isnull().values==False]
# 绘制变换后的直方图和正态概率图：
tmp = np.array(tmp.loc[tmp['TotalBsmtSF']>0, ['TotalBsmtSF']])[:, 0]
# 为啥这里要忽略掉零值呢？做所有样本可发现绘制出的曲线正态性不如之前的。那几个为零的点可以看做是噪声，对正态性产生了不好的影响
# tmp = np.array(tmp.loc[:, ['TotalBsmtSF']])[:, 0]
# 注：series或者dataframe要取出某行某列的话，就用这里提到的loc方法
sns.distplot(tmp, fit=st.norm)
fig = plt.figure()
res = st.probplot(tmp, plot=plt)
# plt.show()


# 同方差性：
# 最好的测量两个变量的同方差性的方法就是图像

# 1、SalePrice 和 GrLivArea 同方差性
tmp = train_test[train_test['SalePrice'].isnull().values==False]
plt.scatter(tmp['GrLivArea'], tmp['SalePrice'])


# 2、SalePrice with TotalBsmtSF 同方差性
tmp = train_test[train_test['SalePrice'].isnull().values==False]
plt.scatter(tmp[tmp['TotalBsmtSF']>0]['TotalBsmtSF'], tmp[tmp['TotalBsmtSF']>0]['SalePrice'])
# 可以看出 SalePrice 在整个 TotalBsmtSF 变量范围内显示出了同等级别的变化
plt.show()


# 三、模型选择


# 1、数据标准化

tmp = train_test[train_test['SalePrice'].isnull().values==False]
tmp_1 = train_test[train_test['SalePrice'].isnull().values==True] # 价格缺失的项

x_train = tmp[['OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
y_train = tmp[["SalePrice"]].values.ravel()
# ravel()将多维数组降为一维，默认为行序优先，[[1,2],[3,4]] to [1, 2, 3, 4]
x_test = tmp_1[['OverallQual', 'GrLivArea','GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]

# 简单测试，用中位数来替代
# print(x_test.GarageCars.mean(), x_test.GarageCars.median(), x_test.TotalBsmtSF.mean(), x_test.TotalBsmtSF.median())

x_test["GarageCars"].fillna(x_test.GarageCars.median(), inplace=True)
x_test["TotalBsmtSF"].fillna(x_test.TotalBsmtSF.median(), inplace=True)
# fillna函数自动将空白处填入 函数内指定的值

