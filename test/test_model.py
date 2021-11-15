#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_model.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/15 11:20  
------------      
"""
from keras import Model, Input

from model import *

# 构建模型
main_input = Input(shape=(5, 2), name='main_input')
enc_output, state_h, state_c = Encoder(hidden_dimensions=256)(main_input)
outputs = Decoder(hidden_dimensions=256)(main_input, enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())

x = tf.random.normal(shape=(5, 5, 2))
y = model(x)
print(y.shape)

main_input = Input(shape=(7, 2), name='main_input')
enc_output, state_h, state_c = Encoder(hidden_dimensions=256)(main_input)
outputs = Decoder(hidden_dimensions=256)(main_input, enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())

x = tf.random.normal(shape=(5, 7, 2))
y = model(x)
print(y.shape)
