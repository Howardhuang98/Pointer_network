#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predict-5to6.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/14 23:44  
------------      
"""
import numpy as np
from keras.layers import Input
from keras.models import Model

from model import *
from utils import cost, prob2rank

x_test = np.random.rand(1, 6, 2)
print(x_test[0])

# 构建模型
main_input = Input(shape=(6, 2), name="6input")
enc_output, state_h, state_c = Encoder(256)(main_input)
outputs = Decoder(256)(main_input, enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())
# 加载权重
model.load_weights(r"./data/checkpoint-2021-11-18-10-09-07.h5", by_name=True, skip_mismatch=True)
y = model.predict(x_test)
model_cost = cost(x_test, y)

print("模型预测路径{}\n最短路径={}".format(prob2rank(y[0]), model_cost[0]))
