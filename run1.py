#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   run1.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/11 9:38  
------------      
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# 加载在TSP_data脚本中生成的数据,位于data文件夹内
X = np.load(r"data/X-1000000.npy")
Y = np.load(r"data/Y-1000000.npy")
YY = np.load(r"data/YY-1000000.npy")
print(X.shape,Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, YY, test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
print(x_test.shape,y_test.shape)

model = tf.keras.models.load_model(r"./data/tmp/checkpoint2021-11-11-00-46-44")
loss, acc = model.evaluate(x_test, y_test,batch_size=128)
print(loss,acc)