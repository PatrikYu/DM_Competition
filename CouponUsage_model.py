#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

# import libraries necessary for this project
import os, sys, pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

from datetime import date

from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
import lightgbm as lgb




# 把数据分析特征工程部分得到的数据传入

from DA import dfoff,dftest,dfon


# 最Naïve的模型

"""
直接用 discount, distance, weekday 类的特征。
train/valid 的划分：用20160101到20160515的作为train，20160516到20160615作为valid。
用线性模型 SGDClassifier

"""
# data split
"""
这里切分的train仅仅在model1中用了一下，后面构造用户和商户特征的时候,用的全是构造出来的feature和data了，后面还会将feature和data
重新合并，并且再次切分为训练样本和交叉验证样本，用作最后的模型评估
"""

df = dfoff[dfoff['label'] != -1].copy() # 对领取了优惠券的用户数据做一个copy
train = df[(df['Date_received'] < '20160516')].copy()
valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
print(train['label'].value_counts())
print(valid['label'].value_counts())

# feature
original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
print(len(original_feature),original_feature)

"""
14 ['discount_rate', 'discount_type', 'discount_man', 'discount_jian', 'distance', 'weekday', 'weekday_type', 
'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7']

"""

# model1
predictors = original_feature
# 构造模型函数 check_model
def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        # SGDClassifier是一系列采用了随机梯度下降（SGD）来求解参数的算法的集合，例如（SVM, logistic regression)等
        # 默认情况下是一个线性（软间隔）支持向量机分类器，大多数SVM的求解使用的是SMO算法，而SVM的另一个解释就是最小化合页损失函数
        # （见李航P113）。因此，该损失函数同样可以通过梯度下降算法来求解参数。
        # 需要注意的是，梯度下降对数据的范围异常敏感，所有要先进行Feature scaling
        # sklearn.linear_model.logistic regression使用的是解析方式来解极大似然问题，而sgdclassifier使用的是梯度下降。
        # 虽然时间上LR要慢于sgd，但是精确度和召回率方面LR更好一些。大规模数据场景下，还是推荐使用梯度下降算法。
        loss='log', # 损失函数选择，默认为hinge即SVM，log为逻辑回归
        penalty='elasticnet', # 惩罚函数为弹性网络，结合了L1和L2先验
        max_iter=100,
        class_weight=None  # 与类相关的权重
    )
    # 利用lambda匿名函数把已确定参数的SGDClassifier封装到classifier中

    model = Pipeline(steps=[
        ('ss', StandardScaler()),
        ('en', classifier())
    ])
    # Pipeline()函数可以把多个“处理数据的节点”按顺序打包在一起，前面的节点都必须实现'fit()'和'transform()'方法，
    # 最后一个节点需要实现fit()方法即可。
    # 先进行特征标准化，再利用SGDClassifier进行分类

    parameters = {'en__alpha': [0.001, 0.01, 0.1] , 'en__l1_ratio': [0.001, 0.01, 0.1]}
                 # en项即 SGDClassifier，参数alpha（惩罚参数），l1_ratio是L1范数前面的值

    folder = StratifiedKFold(n_splits=3, shuffle=True)
    # k折交叉切分，StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    # n_splits：表示划分几等份，shuffle：在每次划分时，是否进行洗牌，若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的

    grid_search = GridSearchCV(     # 超参数自动搜索模块
        model,
        parameters,
        cv=folder,
        n_jobs=-1,
        verbose=1)
    grid_search = grid_search.fit(data[predictors],data['label'])
    return grid_search
    # 自动探索出模型最优的参数，并将对应的SGD模型用下面的代码保存下来


if not os.path.isfile('1_model.pkl'):
    # 若不存在1_model.pkl文件，则执行上面的那个函数，创建一个模型并保存，否则打开已存在的模型
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f) # 将模型保存到1_model.pkl中去
else:
    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f) # 从1_model.pkl中重构为模型


# 预测结果及评价

# 用上面最佳参数模型，在 valid predict 交叉验证集 上进行评估
y_valid_pred = model.predict_proba(valid[predictors])
# predict_proba返回的是一个n行k列的数组，第i行第j列上的数值是模型预测第i个预测样本的标签为j的概率，在这里标签分别为-1,0,1
# predict返回的是一个大小为n的一维数组，一维数组中的第i个值为模型预测第i个预测样本的标签
valid1 = valid.copy()
# Python是地址引用传递，若不适用copy，valid1的改变就等于valid的改变。当然整型那些不会变，list，df什么的都会变
valid1['pred_prob'] = y_valid_pred[:, 1]
# 取出预测为标签0（领取优惠券但未使用）的概率放到'pred_prob'这一列中，即核销概率，这便是我们预测出的核销概率了
valid1.head(2)


# 对每个优惠券coupon_id单独计算核销预测的AUC值（最终评定成绩）

# avgAUC calculation
vg = valid1.groupby(['Coupon_id'])
# 通过优惠券的ID进行分组，行数：不同的优惠券数量，列：所有的属于这个优惠券ID的样本
"""           
优惠券ID1       样本1,4,55...
优惠券ID2       样本2,3,5...54,56...
   ...
"""
aucs = []
for i in vg: # 每次取一种优惠券
    tmpdf = i[1] # 取得此优惠券对应的所有样本
    if len(tmpdf['label'].unique()) != 2: # 即 若label不是2类，就直接跳过，因为AUC无法计算。这样也可以吗？？？
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    # roc_curve(y_test, scores)，y_test为测试集的结果，scores为模型预测的测试集得分（预测输出概率），正标签定为1
    # fpr,tpr,thresholds 分别为不同阈值下的假正率和真正率，阈值：大于阈值认为1
    aucs.append(auc(fpr, tpr)) # 计算此优惠券对应样本的auc值
# 输出此模型在valid数据集上的平均auc值
print(np.average(aucs))
# # 绘制ROC曲线
# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % auc的值) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

# 提交比赛测试样本集合的结果，test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = y_test_pred[:,1] # 核销率
dftest1.to_csv('F:/competition/O2O Coupon Usage Forecast/submit1.csv', index=False, header=False)
dftest1.head()

"""
这个提交结果线上在0.53左右，与valid的结果0.53很接近。这个结果比0.5高一点，但结果算很差。
整个过程中特征是优惠券，距离和时间，没有客户和商户的信息。下面将进行特征的提取。
"""


# 特征提取

"""
通过客户和商户以前的买卖情况，提取各自或者交叉的特征。选择哪个时间段的数据进行特征提取可以进一步研究，
这里使用20160101到20160515之间的数据提取特征，20160516-20160615的数据作为训练集。
"""

feature = dfoff[(dfoff['Date'] < '20160516') | ((dfoff['Date'] == 'null') & (dfoff['Date_received'] < '20160516'))].copy()
# 使用时期在'20160516'之前，或，收到优惠券日期在516之前且没有使用优惠券  的样本 作为 特征提取的来源
# 实际上就是 20160516之前收到优惠券的样本
data = dfoff[(dfoff['Date_received'] >= '20160516') & (dfoff['Date_received'] <= '20160615')].copy()
# 收到优惠券的日期介于20160516-20160615 的样本 作为 训练集
print(data['label'].value_counts())
fdf = feature.copy()

# 用户特征

def userFeature(fdf):
    # key of user
    u = fdf[['User_id']].copy().drop_duplicates()
    # drop_duplicates去除重复项

    # u_coupon_count : num of coupon received by user 统计每个用户收到的优惠券数量（不包含没有优惠券的用户）
    u1 = fdf[fdf['Date_received'] != 'null'][['User_id']].copy()
    u1['u_coupon_count'] = 1
    u1 = u1.groupby(['User_id'], as_index = False).count()

    # u_buy_count : times of user buy offline (with or without coupon)用户在线下购买的次数（包括使用优惠券和不使用优惠券的）
    u2 = fdf[fdf['Date'] != 'null'][['User_id']].copy()
    u2['u_buy_count'] = 1
    u2 = u2.groupby(['User_id'], as_index = False).count()

    # u_buy_with_coupon : times of user buy offline (with coupon)用户在线下使用优惠券购买的次数
    u3 = fdf[((fdf['Date'] != 'null') & (fdf['Date_received'] != 'null'))][['User_id']].copy()
    u3['u_buy_with_coupon'] = 1
    u3 = u3.groupby(['User_id'], as_index = False).count()

    # u_merchant_count : num of merchant user bought from 每个用户购买过的店铺数目
    u4 = fdf[fdf['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
    u4.drop_duplicates(inplace = True) # 去重
    u4 = u4.groupby(['User_id'], as_index = False).count()
    u4.rename(columns = {'Merchant_id':'u_merchant_count'}, inplace = True)
    # rename函数，冒号前为原名称，冒号后为rename后的名称

    # u_min_distance 统计每位用户购买过商品的实体店距离自己的最小距离，最大距离，平均距离以及中位数
    utmp = fdf[(fdf['Date'] != 'null') & (fdf['Date_received'] != 'null')][['User_id', 'distance']].copy()
    # 使用了优惠券购买的用户以及距离
    utmp.replace(-1, np.nan, inplace = True) # 用numpy中的nan替代距离为-1的
    u5 = utmp.groupby(['User_id'], as_index = False).min()
    u5.rename(columns = {'distance':'u_min_distance'}, inplace = True)
    u6 = utmp.groupby(['User_id'], as_index = False).max()
    u6.rename(columns = {'distance':'u_max_distance'}, inplace = True)
    u7 = utmp.groupby(['User_id'], as_index = False).mean()
    u7.rename(columns = {'distance':'u_mean_distance'}, inplace = True)
    u8 = utmp.groupby(['User_id'], as_index = False).median()
    u8.rename(columns = {'distance':'u_median_distance'}, inplace = True)

    # merge all the features on key User_id
    user_feature = pd.merge(u, u1, on = 'User_id', how = 'left')
    # on：用于连接的列名；left：向左合并，即u1合并到u中
    # u-u8的行数可能都是不同的，以行数最大的u为基准，按id合并，有的id号没有相应的属性，自动填充为NaN
    # 注：np.nan是一个特殊的浮点型，None是一个Python特殊的数据类型
    user_feature = pd.merge(user_feature, u2, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u3, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u4, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u5, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u6, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u7, on = 'User_id', how = 'left')
    user_feature = pd.merge(user_feature, u8, on = 'User_id', how = 'left')

    # 每个领取了优惠券的用户使用优惠券购买的比率
    user_feature['u_use_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_coupon_count'].astype('float')
    # 每个购买了的用户使用优惠券购买的比率
    user_feature['u_buy_with_coupon_rate'] = user_feature['u_buy_with_coupon'].astype('float') / user_feature[
        'u_buy_count'].astype('float')

    """
        可以将构造出的商户特征加入原始特征中，再次构造模型，试试看这次得到的交叉验证auc值
        代码见函数外端
    """
    user_feature = user_feature.fillna(0) # 缺失的NaN用常数0填充
    print(user_feature.columns.tolist()) # tolist：将数组或矩阵转换为列表
    return user_feature

"""
# add user feature to data on key User_id
data2 = pd.merge(data, user_feature, on='User_id', how='left').fillna(0)
# split data2 into valid and train
train, valid = train_test_split(data2, test_size=0.2, stratify=data2['label'], random_state=100)
# stratify 保持split前 类的分布，原始数据中两个标签的样本数量之比为4:1，那么切分之后的训练和测试样本中两类比例依然为4:1
# 通常在这种类分布不平衡的情况下会用到stratify。
# random_state随机数种子，实际上就是该组随机数的编号，编号相同，每次都会得到一组相同的随机数，填0或不填每次得到的随机数组不同

# model2
predictors = original_feature + user_feature.columns.tolist()[1:]
# predictors是 ['discount_rate','discount_type','discount_man'...]
print(len(predictors), predictors)
if not os.path.isfile('2_model.pkl'):
    model = check_model(train, predictors)
    print(model.best_score_)
    print(model.best_params_)
    with open('2_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('2_model.pkl', 'rb') as f:
        model = pickle.load(f)

# valid set performance
valid['pred_prob'] = model.predict_proba(valid[predictors])[:, 1]
validgroup = valid.groupby(['Coupon_id'])
aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
    # aucs.append(roc_auc_score(tmpdf['label'], tmpdf['pred_prob']))
print(np.average(aucs))

# 结果为0.598，有所进步
"""




# 商户特征

def merchantFeature(df):    # df是之前取得的特征样本
    m = df[['Merchant_id']].copy().drop_duplicates()

    # m_coupon_count : num of coupon from merchant 每个商户被领取的优惠券数目
    m1 = df[df['Date_received'] != 'null'][['Merchant_id']].copy()
    m1['m_coupon_count'] = 1
    m1 = m1.groupby(['Merchant_id'], as_index=False).count()

    # m_sale_count : num of sale from merchant (with or without coupon) 商户被购买的次数（包括使用与未使用优惠券的购买）
    m2 = df[df['Date'] != 'null'][['Merchant_id']].copy()
    m2['m_sale_count'] = 1
    m2 = m2.groupby(['Merchant_id'], as_index=False).count()

    # m_sale_with_coupon : num of sale from merchant with coupon usage 商户被使用优惠券购买的次数
    m3 = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['Merchant_id']].copy()
    m3['m_sale_with_coupon'] = 1
    m3 = m3.groupby(['Merchant_id'], as_index=False).count()

    # m_min_distance 距离每个商户最近的用户距离，最大距离，平均距离，中位数
    mtmp = df[(df['Date'] != 'null') & (df['Date_received'] != 'null')][['Merchant_id', 'distance']].copy()
    mtmp.replace(-1, np.nan, inplace=True)
    m4 = mtmp.groupby(['Merchant_id'], as_index=False).min()
    m4.rename(columns={'distance': 'm_min_distance'}, inplace=True)
    m5 = mtmp.groupby(['Merchant_id'], as_index=False).max()
    m5.rename(columns={'distance': 'm_max_distance'}, inplace=True)
    m6 = mtmp.groupby(['Merchant_id'], as_index=False).mean()
    m6.rename(columns={'distance': 'm_mean_distance'}, inplace=True)
    m7 = mtmp.groupby(['Merchant_id'], as_index=False).median()
    m7.rename(columns={'distance': 'm_median_distance'}, inplace=True)

    merchant_feature = pd.merge(m, m1, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m2, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m3, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m4, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m5, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m6, on='Merchant_id', how='left')
    merchant_feature = pd.merge(merchant_feature, m7, on='Merchant_id', how='left')

    # 每个商户被领取优惠券的使用率
    merchant_feature['m_coupon_use_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / merchant_feature[
        'm_coupon_count'].astype('float')
    # 每个商户卖出去商品中通过优惠券购买的比率
    merchant_feature['m_sale_with_coupon_rate'] = merchant_feature['m_sale_with_coupon'].astype('float') / \
                                                  merchant_feature['m_sale_count'].astype('float')

    """
    可以将构造出的商户特征加入上次构造出的特征中（已包含用户特征），再次构造模型即为model3，试试看这次得到的交叉验证auc值
    代码与之前的类似，最终结果显示，这次的auc值达到了0.602，再一次上升
    """
    merchant_feature = merchant_feature.fillna(0)
    print(merchant_feature.columns.tolist())
    return merchant_feature



def usermerchantFeature(df):

    um = df[['User_id', 'Merchant_id']].copy().drop_duplicates()

    # 统计用户与同一个商户产生“联系”的次数（可能领取了优惠券或有购买行为）
    um1 = df[['User_id', 'Merchant_id']].copy()
    um1['um_count'] = 1
    um1 = um1.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    # 统计使用了优惠券的用户在同一个商户购买的次数
    um2 = df[df['Date'] != 'null'][['User_id', 'Merchant_id']].copy()
    um2['um_buy_count'] = 1
    um2 = um2.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    # 统计领取了优惠券的用户在同一个商户购买的次数
    um3 = df[df['Date_received'] != 'null'][['User_id', 'Merchant_id']].copy()
    um3['um_coupon_count'] = 1
    um3 = um3.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    # 统计领取了优惠券并使用此优惠券购买了的用户在同一个商户购买的次数
    um4 = df[(df['Date_received'] != 'null') & (df['Date'] != 'null')][['User_id', 'Merchant_id']].copy()
    um4['um_buy_with_coupon'] = 1
    um4 = um4.groupby(['User_id', 'Merchant_id'], as_index = False).count()

    user_merchant_feature = pd.merge(um, um1, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um2, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um3, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = pd.merge(user_merchant_feature, um4, on = ['User_id','Merchant_id'], how = 'left')
    user_merchant_feature = user_merchant_feature.fillna(0)

    # 在同一个商户处用户购买的比率
    user_merchant_feature['um_buy_rate'] = \
        user_merchant_feature['um_buy_count'].astype('float')/user_merchant_feature['um_count'].astype('float')
    # 在同一个商户处领取了优惠券，使用了优惠券的比率
    user_merchant_feature['um_coupon_use_rate'] = \
        user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_coupon_count'].astype('float')
    # 在同一个商户处购买了，使用了优惠券的比率
    user_merchant_feature['um_buy_with_coupon_rate'] = \
        user_merchant_feature['um_buy_with_coupon'].astype('float')/user_merchant_feature['um_buy_count'].astype('float')
    """
       可以将构造出的特征加入上次构造出的特征中（已包含用户和商户特征），再次构造模型即为model4，试试看这次得到的交叉验证auc值
       代码与之前的类似，最终结果显示，这次的auc值达到了0.6157，再一次上升
    """
    user_merchant_feature = user_merchant_feature.fillna(0)
    print(user_merchant_feature.columns.tolist())
    return user_merchant_feature

# 将此前构造的特征放入

"""
feature = dfoff[(dfoff['Date'] < '20160516') | ((dfoff['Date'] == 'null') & (dfoff['Date_received'] < '20160516'))].copy()
data = dfoff[(dfoff['Date_received'] >= '20160516') & (dfoff['Date_received'] <= '20160615')].copy()
data是特征提取的训练集，衡量增加了新特征之后效果如何。
feature提取的是 20160516之前收到优惠券的样本，比data和dftest的样本数量要多，而且很有可能很多用户和商户的ID不重合，
从下面merge函数可以看出，一律按照test中处理，多出来的就丢弃，没有的就填NaN（然后用0替换）。
之前的feature没有改变，因为上面的处理都copy到fdf中去处理了，下面函数的参数可以直接传入。

"""

def featureProcess(feature, train, test):   # feature, data, dftest

    user_feature = userFeature(feature)
    merchant_feature = merchantFeature(feature)
    user_merchant_feature = usermerchantFeature(feature)

    train = pd.merge(train, user_feature, on='User_id', how='left') # 相当于把feature和data又重新组合成了原始的数据集了
    # 左连接，左侧DataFrame取全部，右侧DataFrame取部分
    train = pd.merge(train, merchant_feature, on='Merchant_id', how='left')
    train = pd.merge(train, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    train = train.fillna(0)

    test = pd.merge(test, user_feature, on='User_id', how='left')
    test = pd.merge(test, merchant_feature, on='Merchant_id', how='left')
    test = pd.merge(test, user_merchant_feature, on=['User_id', 'Merchant_id'], how='left')
    test = test.fillna(0)

    return train, test

# feature engineering
train, test = featureProcess(feature, data, dftest)

# features
predictors = ['discount_rate', 'discount_man', 'discount_jian', 'discount_type', 'distance',
              'weekday', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6',
              'weekday_7', 'weekday_type',
              'u_coupon_count', 'u_buy_count', 'u_buy_with_coupon', 'u_merchant_count', 'u_min_distance',
              'u_max_distance', 'u_mean_distance', 'u_median_distance', 'u_use_coupon_rate', 'u_buy_with_coupon_rate',
              'm_coupon_count', 'm_sale_count', 'm_sale_with_coupon', 'm_min_distance', 'm_max_distance',
              'm_mean_distance', 'm_median_distance', 'm_coupon_use_rate', 'm_sale_with_coupon_rate', 'um_count', 'um_buy_count',
              'um_coupon_count', 'um_buy_with_coupon', 'um_buy_rate', 'um_coupon_use_rate', 'um_buy_with_coupon_rate']
print(len(predictors), predictors)

# 将训练样本再切分为 训练集和交叉验证集
trainSub, validSub = train_test_split(train, test_size = 0.2, stratify = train['label'], random_state=100)



# 建模啦


# 线性模型

model = check_model(trainSub, predictors)

validSub['pred_prob'] = model.predict_proba(validSub[predictors])[:,1]
validgroup = validSub.groupby(['Coupon_id'])
aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))


# GBM lightgbm

model = lgb.LGBMClassifier(
                    learning_rate = 0.01,
                    boosting_type = 'gbdt', # 提升树的类型 gbdt,dart,goss,rf
                    objective = 'binary',
                    metric = 'logloss',
                    max_depth = 5,  # 最大的树深度
                    sub_feature = 0.7,
                    num_leaves = 3, # 树的最大叶子树
                    colsample_bytree = 0.7, # 训练特征采样率 列
                    n_estimators = 5000, # 拟合的树的棵树，相当于训练轮数
                    early_stop = 50,
                    verbose = -1
                    )
model.fit(trainSub[predictors], trainSub['label'])

validSub['pred_prob'] = model.predict_proba(validSub[predictors])[:,1]
validgroup = validSub.groupby(['Coupon_id'])
aucs = []
for i in validgroup:
    tmpdf = i[1]
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))

# test prediction for submission
y_test_pred = model.predict_proba(test[predictors])
submit = test[['User_id','Coupon_id','Date_received']].copy()
submit['label'] = y_test_pred[:,1]
submit.to_csv('F:/competition/O2O Coupon Usage Forecast/submit2.csv', index=False, header=False)
submit.head()

"""
这个提交结果线上在0.60左右，跟最好的结果0.80还有很大差距

可以探索的点：
1、数据的划分。
2、更多的特征。
3、模型（xgboost，lightgbm，gbdt）的尝试。

拓展思路：https://github.com/wepe/O2O-Coupon-Usage-Forecast

赛题提供的预测集中，包含了同一个用户在整个7月份里的优惠券领取情况，这实际上是一种leakage，比如存在这种情况：
某一个用户在7月10日领取了某优惠券，然后在7月12日和7月15日又领取了相同的优惠券，那么7月10日领取的优惠券被核销的可能性就很大了。
我们在做特征工程时也注意到了这一点，提取了一些相关的特征。加入这部分特征后，AUC提升了10个百分点

"""
