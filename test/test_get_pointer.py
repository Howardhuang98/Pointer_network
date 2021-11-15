#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_get_pointer.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/15 12:28  
------------      
"""
from model import *

x = tf.random.normal(shape=(5,5,2))
probs = tf.random.normal(shape=(5,5))
points = _get_pointer(x,probs)
print(x)
print(points)