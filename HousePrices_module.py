#  coding: utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')      # python的str默认是ascii编码，和unicode编码冲突,需要加上这几句

from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    # 设置作图中显示中文字体

# bagging:

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso,ElasticNet
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge




from DA import x_train,y_train,x_test
import pytest

####!!!!!!!!!!!! 重要问题：DA.py是根据官方教程写的，实际上有些步骤被省略了，数据分析部分参考
# https://blog.csdn.net/Irving_zhang/article/details/78561105
# DA.py 上的做图分析方法可以作为参考！应当以csdn上文章为主。

ridge = Ridge(alpha=0.1)
# bagging 把很多小的分类器放在一起，每个train随机的一部分数据，然后把它们的最终结果综合起来（多数投票）
# bagging 算是一种算法框架
params = [1, 5, 10, 15, 20, 25, 30, 35, 40, 60]
test_scores = []
for param in params:
    clf = BaggingRegressor(base_estimator=ridge, n_estimators=param)
    # base_estimator默认是DecisionTree，n_estimators是要集成的基估计器的个数
    test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    # cv=5表示cross_val_score采用的是k-fold cross validation的方法，重复5次交叉验证
    # 将数据集分为10折，做一次交叉验证，实际上它是计算了十次，将每一折都当做一次测试集，其余九折当做训练集，这样循环十次
    # scoring='precision'、scoring='recall'、scoring='f1', scoring='neg_mean_squared_error' 方差值
    test_scores.append(np.mean(test_score)) # 取十次得到的test_score取平均值

print(test_score.mean()) # 不同基估计器得到的正确率平均值
# fig = plt.figure()
plt.figure(15) # # 把接下来的图显示为figure(0)
plt.plot(params, test_scores)
plt.title('n_estimators vs CV Error')
plt.show()



# 模型选择
## LASSO Regression (L2正则线性回归):
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
# Pipeline可以将许多算法模型串联起来，如将特征提取、归一化、分类或回归在一起形成一个典型的机器学习问题工作流
# RobustScaler()对有离群点的数据进行归一化效果更好，通常归一化到0到1之间
# The Lasso 是估计稀疏系数的线性模型。α和SVM的正则化参数C:α=1/C,alpha=0就相当于最小二乘法，random_state是产生随机特征的个数
# Elastic Net Regression（弹性网络回归） :

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
# 弹性网络回归是岭回归与Lasso回归的结合，l1_ratio就是正则化的参数lambda（L1正则化参数），等于1时相当于LASSO
# 第二个则是L2正则化参数，elastic net在具有多个特征，并且特征之间具有一定关联的数据中比较有用
# L1损失和L2损失只是MAE和MSE的别称，处理异常点时，L1损失函数更稳定，但它的导数不连续，因此求解效率较低。
# L2损失函数对异常点更敏感，但通过令其导数为0，可以得到更稳定的封闭解
## Kernel Ridge Regression
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
# degree是poly核中的参数d（度），coef0是poly和sigmoid核中的0参数的替代值
## Gradient Boosting Regression(梯度提升回归)
GBoost = GradientBoostingRegressor(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=4,
    max_features='sqrt',
    min_samples_leaf=15,
    min_samples_split=10,
    loss='huber',
    random_state=5)
# GBR是一种基础学习算法，n_estimators，指GBR使用的学习算法的数量，每个学习算法都是一颗决策树，max_depth决定了树生成的节点数
# loss参数决定损失函数，ls是默认值，表示最小二乘法（least squares）
# Huber是平滑的平均绝对误差MAE，在误差很小时会变为平方误差MSE，误差降到多小时变为二次误差由超参数δ（delta）来控制
# 使用MAE训练神经网络最大的一个问题就是不变的大梯度，这可能导致在使用梯度下降快要结束时，错过了最小点。
# 而对于MSE，梯度会随着损失的减小而减小，使结果更加精确。
# 在这种情况下，Huber损失就非常有用。它会由于梯度的减小而落在最小值附近。比起MSE，它对异常点更加鲁棒。
# 因此，Huber损失结合了MSE和MAE的优点。但是，Huber损失的问题是我们可能需要不断调整超参数delta。
## XGboost
import xgboost as xgb
model_xgb = xgb.XGBRegressor(
    colsample_bytree=0.4603,
    gamma=0.0468,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,
    n_estimators=2200,
    reg_alpha=0.4640,
    reg_lambda=0.8571,
    subsample=0.5213,
    silent=1,
    random_state=7,
    nthread=-1)
# 在XGBoost里，每棵树是一个一个往里面加的,需要保证加上新的树之后，目标函数（就是损失）的值会下降,并且要限制叶子结点的个数
# 以防止过拟合。
## lightGBM
import lightgbm as lgb
model_lgb = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=5,
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,
    bagging_fraction=0.8,
    bagging_freq=5,
    feature_fraction=0.2319,
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,
    min_sum_hessian_in_leaf=11)
# LightGBM （Light Gradient Boosting Machine）是一个实现 GBDT 算法的框架，支持高效率的并行训练，
# 默认的训练决策树时使用直方图算法，这是一种牺牲了一定的切分准确性而换取训练速度以及节省内存空间消耗的算法
# GBDT 在每一次迭代的时候，都需要遍历整个训练数据多次，面对工业级海量的数据，普通的 GBDT 算法是不能满足其需求的
# 注：GBDT (Gradient Boosting Decision Tree)其主要思想是利用弱分类器（决策树）迭代训练以得到最优模型，
# 该模型具有训练效果好、不易过拟合等优点。

## 对这些基本模型进行打分

# 我们使用Sklearn的cross_val_score函数。然而这个函数没有shuffle方法，我们添加了一行代码，为了在交叉验证之前shuffle数据集。
#Validation function

def rmsle_cv(model):
    test_score = np.sqrt(-cross_val_score(model, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    return(test_score)
#
# n_folds = 10
# def rmsle_cv(model):
#     kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
#     rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
#     return(rmse)


score = rmsle_cv(lasso)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print(
    "Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(),
                                                          score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


train_sizes, train_loss, test_loss = learning_curve(ridge, x_train, y_train, cv=10,
                                                    scoring='neg_mean_squared_error',
                                                    train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9 , 0.95, 1])

# 训练误差均值
train_loss_mean = -np.mean(train_loss, axis = 1)
# 测试误差均值
test_loss_mean = -np.mean(test_loss, axis = 1)

# 绘制误差曲线
plt.plot(train_sizes/len(x_train), train_loss_mean, 'o-', color = 'r', label = 'Training')
plt.plot(train_sizes/len(x_train), test_loss_mean, 'o-', color = 'g', label = 'Cross-Validation')

plt.xlabel('Training data size')
plt.ylabel('Loss')
plt.legend(loc = 'best')
plt.show()


# 四、建立模型

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# Stacking模型融合：

# Average-Stacking：
# 我们从最简单的平均基本模型的Stacking方法开始模型融合。建立一个新的类来扩展scikit模型融合方法：

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack( [model.predict(X) for model in self.models_] )
        return np.mean(predictions, axis=1)

# 平均四个模型ENet，GBoost，KRR和lasso。利用上面重写的方法，我们可以轻松地添加更多的模型：
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f}) \n".format(score.mean(),score.std()))

# 可以看到，均方误差比单独使用几个模型有所下降，这还是最简单的模型融合，这鼓励我们向着更深的模型融合的方向继续努力。

# Meta-model Stacking：

# 算法：
#
# 1、将整个训练集分解成两个不相交的集合（这里是train和.holdout）。
# 2、在第一部分（train）上训练几个基本模型。
# 3、在第二个部分（holdout）上测试这些基本模型。
# 4、使用(3)中的预测（称为 out-of-fold 预测）作为输入，并将正确的标签（目标变量）作为输出来训练更高层次的学习模型称为元模型。
#
# 前三个步骤是迭代完成的。例如，如果我们采取5倍的fold，我们首先将训练数据分成5次。然后我们会做5次迭代。
# 在每次迭代中，我们训练每个基础模型4倍，并预测剩余的fold（holdout fold）。

from sklearn.model_selection import KFold

class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    #Do the predictions of all base models on the test data and use the averaged predictions as
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

# 测试Meta-model Stacking结果：

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),meta_model = lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# 我们得到了一个更好的结果


# 最后，为了得到最后提交的结果，我们将StackedRegressor、XGBoost和LightGBM进行融合，得到rmsle的结果。

from sklearn.metrics import mean_squared_error
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# StackedRegressor:
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))

# XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))

# lightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''

print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15))

# 模型融合的预测效果
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

# 保存结果
result = pd.DataFrame()
result['Id'] = test_ID
result['SalePrice'] = ensemble
# index=False 是用来除去行编号
result.to_csv('/Users/liudong/Desktop/house_price/result.csv', index=False)
