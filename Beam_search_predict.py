#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   Beam_search_predict.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/18 19:50  
------------      
"""
import numpy as np
from keras.layers import Input
from keras.models import Model

from model import *
from utils import cost, prob2rank

x_test = np.load(r"tsp_data/tsp_5/tsp5_test_x.npy")
y_test = np.load(r"tsp_data/tsp_5/tsp5_test_y.npy")

# 构建模型
main_input = Input(shape=(x_test.shape[1], 2), name='main_input')
enc_output, state_h, state_c = Encoder(256)(main_input)
outputs = Beam_decoder(256)(main_input, enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())
# 指定训练配置
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 加载权重
model.load_weights(r"./data/checkpoint-2021-11-18-10-09-07.h5")
print(x_test.shape, y_test.shape)
y = model.predict(x_test)
model_cost = cost(x_test, y)
optimal_cost = cost(x_test, y_test)

print("模型预测路径长度-最短路径=", sum(model_cost - optimal_cost) / x_test.shape[0])
print(prob2rank(y[320]), prob2rank(y_test[320]))
print(prob2rank(y[20]), prob2rank(y_test[20]))
print(prob2rank(y[5]), prob2rank(y_test[5]))
print(prob2rank(y[100]), prob2rank(y_test[100]))
print(prob2rank(y[1000]), prob2rank(y_test[1000]))
