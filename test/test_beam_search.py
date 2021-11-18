#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_beam_search.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/17 11:19  
------------      
"""

from model import *

x = tf.random.normal(shape=(200, 5, 2))
encoder_outputs = tf.random.normal(shape=(200, 5, 128))
state = [tf.random.normal(shape=(200, 128)), tf.random.normal(shape=(200, 128))]
dec = Beam_decoder()
y = dec(x, encoder_outputs, state)
print(y.shape)
