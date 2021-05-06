# -*- coding: utf-8 -*-
"""
Spyder Editor
matilda
20200610
This is a temporary script file.
"""

#查看sklearn中的所有的模型评估指标
import sklearn
sorted(sklearn.metrics.SCORERS.keys())

#boston data on xgboost
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor as xgbr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LinearRegression as linear
from sklearn.model_selection import KFold,cross_val_score as cvs,train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import load_boston
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from time import time
import datetime 


data = load_boston()
x = data.data
y = data.target

xtrain,xtest,ytrain,ytest = tts(x,y,test_size = 0.3,random_state = 420)
reg = xgbr(n_setimators = 100).fit(xtrain,ytrain)
#预测值&score
reg.predict(xtest)
reg.score(xtest,ytest)
# 均方误差和均值
mse(ytest,reg.predict(xtest))
y.mean()
#特征的重要性
reg.feature_importances_

#3.以下用交叉验证来对比,xgbr 随机森林，线性回归
reg = xgbr(n_eatimators = 100)
cvs(reg,xtrain,ytrain,cv = 5 ).mean() # 交叉验证
cvs(reg,xtrain,ytrain,cv = 5 ,scoring = 'neg_mean_squared_error').mean()

rfr = rfr(n_estimators = 100)
cvs(rfr,xtrain,ytrain,cv = 5).mean()
cvs(reg,xtrain,ytrain,cv = 5 ,scoring = 'neg_mean_squared_error').mean()

lr = linear()
cvs(rfr,xtrain,ytrain,cv = 5).mean()
cvs(reg,xtrain,ytrain,cv = 5 ,scoring = 'neg_mean_squared_error').mean()

#4.定义绘制以训练样本数为横坐标的学习函数
def plot_learning_curve(estimator,title,x,y
                        ,ax = None
                        ,ylim = None
                        ,cv = None
                        ,n_jobs = None
                        ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_size,train_scores,test_scores = learning_curve(estimator,x,y
                                                         ,shuffle = True
                                                         ,cv = cv
                                                         ,random_state = 420
                                                         ,n_jobs = n_jobs)
    if ax ==None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel('training examples')
    ax.set_ylabel('score')
    ax.grid()
    ax.plot(train_size,np.mean(train_scores,axis =1),marker='o', mec='r', mfc='w',color = 'r',label = 'training score')
    ax.plot(train_size,np.mean(test_scores,axis =1), 'o--',color = 'g',label = 'test score')
    ax.legend(loc = 'best')
    return ax
#画图
cv = KFold(n_splits = 5, shuffle = True, random_state = 420)
plot_learning_curve(xgbr(objective ='reg:squarederror', n_estimators=100, random_state=420), 'xgb', xtrain, ytrain, ax = None, cv =cv )
plt.show()                        
'''发现出现过拟合，下面调整参数                        '''


#6，使用参数学习曲线观察n_estimators对模型的影响
axisx = range(10,1010,50)
rs = []
for i in axisx:
    reg = xgbr(objective ='reg:squarederror', n_estimators = i,randoms = 420)
    rs.append(cvs(reg,xtrain,ytrain,cv = cv).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure() 
plt.plot(axisx,rs,c = 'g',label= 'xgb')
plt.legend()
plt.show()
''' 返回最优点在460 达到83.79% 同时也会发现在30左右就已经达到80%，此后曲线趋于平缓 
    （其中objective ='reg:squarederror' 是为了让其不显示warning）
'''                           

#7，画特殊的学习曲线
axis = range(100,300,10)
rs = []
var = []
ge = []
for i in axis:
    reg = xgbr(objective ='reg:squarederror',n_eatimators = i,random = 420) 
    cvresult = cvs(reg, xtrain, ytrain, cv = cv)  
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1-cvresult.mean())**2+cvresult.var())         
print(axis[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])
print(axis[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axis[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))], min(ge))                        
rs = np.array(rs)
var = np.array(var)*0.01
#ge = np.array(ge)
plt.figure()
plt.plot(axis,rs,c='red',label = 'xgb')
plt.plot(axis,rs+var,c='black',linestyle ='-.' )
plt.plot(axis,rs-var,c='black',linestyle ='-.' )
plt.legend()
plt.show()
'''
rs 记录了1-偏差； var记录方差； ge计算泛化误差的可控部分；
第一次print：R2最高对应的参数取值，并打印这个参数下的方差
第二次print：方差最低时对应的参数取值，并打印这个参数下的R2
第三次print：泛化误差可控部分的参数取值，并打印这个参数下的R2,var，及泛化误差的可控部分 

'''                        
'''
up wrong

'''

# 验证效果
#time0= time
print(xgbr(n_estimators = 100, random = 420).fit(xtrain,ytrain).score(xtest,ytest))
#print(time()-time0)
print(xgbr(n_estimators = 660, random = 420).fit(xtrain,ytrain).score(xtest,ytest))
print(xgbr(n_estimators = 180, random = 420).fit(xtrain,ytrain).score(xtest,ytest))






























                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
    



