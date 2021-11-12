#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   predict.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/11 9:38  
------------      
"""

import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.utils.np_utils import to_categorical

from utils import Tsp
from model import *

# 生成问题
p = Tsp()
X, Y = p.next_batch(2)
YY = []
for y in Y:
    YY.append(to_categorical(y))
YY = np.asarray(YY)
print(X,YY)
# 构建模型
main_input = Input(shape=(X.shape[1], 2), name='main_input')
enc_output, state_h, state_c = Encoder()(main_input)
outputs = Decoder()(enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())
# 加载权重
model.load_weights(r"data/tmp/checkpoint-2021-11-11-20-56-45")
y = model.predict(X)
y = y.argmax(axis=2)
p.cost(X,y)
print(p.cost(X,y),p.cost(X,Y))