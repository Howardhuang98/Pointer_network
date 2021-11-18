#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_attention.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/15 10:47  
------------      
"""
from model import *

encoder_outputs, dec_output = tf.random.normal(shape=(5, 10, 2)), tf.random.normal(shape=(5, 128))
attention = Attention()
y = attention(encoder_outputs, dec_output)
print(y.shape)
