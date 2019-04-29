#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

"""
https://tianchi.aliyun.com/notebook/detail.html?spm=5176.11409386.4851167.7.65c91d07FiVHVN&id=4796

"""
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


# 数据分析


# 读取数据以及简单分析

dfoff = pd.read_csv('F:/competition/O2O Coupon Usage Forecast/ccf_offline_stage1_train.csv')
dftest = pd.read_csv('F:/competition/O2O Coupon Usage Forecast/ccf_offline_stage1_test_revised.csv')

dfon = pd.read_csv('F:/competition/O2O Coupon Usage Forecast/ccf_online_stage1_train.csv')

dfoff.info()

print('有优惠券，购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
print('无优惠券，购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
print('有优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
print('无优惠券，不购买商品条数', dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])

# 在测试集中出现的用户但训练集没有出现
print('1. User_id in training set but not in test set', set(dftest['User_id']) - set(dfoff['User_id']))
# 在测试集中出现的商户但训练集没有出现
print('2. Merchant_id in training set but not in test set', set(dftest['Merchant_id']) - set(dfoff['Merchant_id']))


# 优惠券与距离

print('Discount_rate 类型:',dfoff['Discount_rate'].unique())
print('Distance 类型:', dfoff['Distance'].unique())
# unique输出所有不同的值

# convert Discount_rate and Distance

def getDiscountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row: # 满减，type=1，两种活动可能对用户使用优惠券的吸引力有所不同
        return 1
    else:            # 打折，type=0
        return 0


def convertRate(row):
    """Convert discount to rate 将满减或None转换为打折率"""
    if row == 'null':
        return 1.0 # 折率1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1]) / float(rows[0]) # 这种折算不准，但是有一定参考价值
    else:
        return float(row)


def getDiscountMan(row): # 将满减需要达到的金额提取出来作为一个特征，不属于此类型的定为0
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0


def getDiscountJian(row): # 将满减能优惠的金额提取出来作为一个特征，不属于此类型的定为0
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0


def processData(df):
    # convert discunt_rate
    # 通过apply的方式对于某一列的所有值进行函数处理
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)

    print(df['discount_rate'].unique())

    # convert distance
    # 下面的程序实现了两个功能，首先将没有距离信息的值替换为-1，然后将所有值的字符型转换为int
    df['distance'] = df['Distance'].replace('null', -1).astype(int)
    print(df['distance'].unique())
    return df

dfoff = processData(dfoff)
dftest = processData(dftest)


# 时间

# 列出所有收到优惠券的不重复时间并对非空值进行排序
date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[date_received != 'null'])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[date_buy != 'null'])

print('优惠券收到日期从',date_received[0],'到', date_received[-1])
print('消费日期从', date_buy[0], '到', date_buy[-1])

# 看一下每天的顾客收到coupon的数目，以及收到coupon后用coupon消费的数目

# 收到优惠券的数目
couponbydate = dfoff[dfoff['Date_received'] != 'null'][['Date_received', 'Date']]\
               .groupby(['Date_received'], as_index=False).count()
# groupby函数根据Date_received分组，并通过count计数，组数为收到优惠券日期的个数
# 注意：这里的每一组都是每一行，一共只有两列，是['Date_received','count']
couponbydate.columns = ['Date_received','count']
# 收到优惠券并使用的数目
buybydate = dfoff[(dfoff['Date'] != 'null') & (dfoff['Date_received'] != 'null')][['Date_received', 'Date']]\
            .groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received','count']

sns.set_style('ticks')
sns.set_context("notebook", font_scale= 1.4)
plt.figure(figsize = (12,8))
date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')
# 将日期变为 年-月-日 的格式

# 作条形图
plt.subplot(211)
plt.bar(date_received_dt, couponbydate['count'], label = 'number of coupon received' )
plt.bar(date_received_dt, buybydate['count'], label = 'number of coupon used')
plt.yscale('log') # 纵坐标采用对数刻度
plt.ylabel('Count')
plt.legend()

plt.subplot(212)
plt.bar(date_received_dt, buybydate['count']/couponbydate['count']) # 使用优惠券的比例
plt.ylabel('Ratio(coupon used/coupon received)')
plt.tight_layout() # 图像外部边缘的调整可以使用plt.tight_layout()进行自动控制

# 新建关于星期的特征

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1
            #  date(year, month, day)，weekday()函数返回的是当前日期所在的星期数，返回的0-6代表周一--到周日

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# 新建 weekday_type 这一特征 :  周六和周日为1，其他为0。
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )
dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0 )

# change weekday to one-hot encoding
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
print(weekdaycols)
# 输出： ['weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 'weekday_6', 'weekday_7']

# 将null值替换为numpy中的nan值，并根据weekday这一特征进行分列
tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols # 列的名称即weekdaycols
dfoff[weekdaycols] = tmpdf # 将新的列加入dfoff中

tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


# 数据标注

"""
Date_received == 'null': y = -1
Date != 'null' & Date-Date_received <= 15: y = 1
Otherwise: y = 0

"""
def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'): # 后面的'D'代表天，若在十五天之内使用了，记为1
            return 1
    return 0
dfoff['label'] = dfoff.apply(label, axis = 1) # dfoff执行label函数完成标注
print(dfoff['label'].value_counts())

"""
 0    988887

-1    701602

 1     64395

"""

print('已有columns：',dfoff.columns.tolist())

"""
['User_id', 'Merchant_id', 'Coupon_id', 'Discount_rate', 'Distance', 'Date_received', 'Date', 'discount_rate', 
'discount_man', 'discount_jian', 'discount_type', 'distance', 'weekday', 'weekday_type', 'weekday_1', 'weekday_2', 
'weekday_3', 'weekday_4','weekday_5', 'weekday_6', 'weekday_7', 'label']

"""