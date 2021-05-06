#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 16:34:19 2020

@author: matlida
"""


from sklearn import tree 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

wine = load_wine()
wine.data
wine.target
import pandas as pd 
wine.feature_names
wine.target_names

xtrain,xtest,ytrain,ytest = train_test_split(wine.data,wine.target,
                                             test_size =0.3)
print(xtrain.shape)
print(ytrain.shape)
clf = tree.DecisionTreeClassifier(criterion = 'entropy',random_state = 30)
clf.fit(xtrain,ytrain)
score = clf.score(xtest,ytest)
print(score)

feature_names = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮',
                 '非黄烷类酚类','花青素','颜 色强度','色调',
                 'od280/od315稀释葡萄酒','脯氨酸']
import graphviz
dot_data = tree.export_graphviz(clf,feature_names = feature_names
                                ,class_names = ["琴酒","雪莉","贝尔摩德"]
                                ,filled = True
                                ,rounded = True)
graph = graphviz.Source(dot_data)



























