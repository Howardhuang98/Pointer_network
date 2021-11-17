#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test_beam_search.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/17 11:19  
------------      
"""
import numpy as np
from model import *
import tensorflow_addons as tfa

from model import *

x = tf.random.normal(shape=(128,5,2))
probs = tf.random.normal(shape=(128,5))

y0 = get_pointer(x, probs)


print()
