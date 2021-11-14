#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predict.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/12 16:58  
------------      
"""
import numpy as np
from keras.layers import Input
from keras.models import Model

from model import *
from utils import cost

x_test = np.load(r"tsp_data/tsp_5/tsp5_test_x.npy")
y_test = np.load(r"tsp_data/tsp_5/tsp5_test_y.npy")

# 构建模型
main_input = Input(shape=(x_test.shape[1], 2), name='main_input')
enc_output, state_h, state_c = Encoder(256)(main_input)
outputs = Decoder(256)(enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())
# 指定训练配置
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 加载权重
model.load_weights(r"./data/ckp-2021-11-13-10-20-37/checkpoint")
print(x_test.shape,y_test.shape)
y = model.predict(x_test)
model_cost = cost(x_test, y)
optimal_cost = cost(x_test, y_test)

print("模型预测路径长度-最短路径=", sum(model_cost - optimal_cost) / x_test.shape[0])
