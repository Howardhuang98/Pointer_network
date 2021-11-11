#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model.py    
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/9 15:26  
------------      
"""
import keras
import keras.backend as K
from keras.activations import tanh, softmax
from keras.engine.base_layer import InputSpec
from keras.layers import LSTM
import tensorflow as tf


class Encoder(keras.layers.Layer):
    """
    编码器
    """

    def __init__(self, hidden_dimensions=128):
        """
        :param hidden_dimensions: 指定隐藏状态 h 的维度
        """
        super(Encoder, self).__init__(name='encoder', trainable=True)
        self.lstm = LSTM(units=hidden_dimensions, return_sequences=True, return_state=True, name="encoder")

    def call(self, x):
        """

        :param x: 输入数据 [batch, time steps, feature]
        :return:
        enc_output: [h1,h2,h3...h_n]
        state_h: h_n
        state_c: c_n
        """
        enc_output, state_h, state_c = self.lstm(x)
        return enc_output, state_h, state_c


class DecoderCell(keras.layers.LSTMCell):
    """
    解码器，要求输入为[batch,features]
    """

    def __init__(self, hidden_dimensions=128):
        super(DecoderCell, self).__init__(units=hidden_dimensions)

    def call(self, inputs, states):
        """
        :param inputs: [batch,features]
        :param states: [h,c] 两个张量的列表，每个张量的形状为 [batch, units]
        :return:
        """
        state_h, [state_h, state_c] = super(DecoderCell, self).call(inputs, states)
        return state_h, [state_h, state_c]


class Attention(keras.layers.Layer):
    """
    注意力层
    """

    def __init__(self, hidden_dimensions=128, name='attention'):
        super(Attention, self).__init__(name=name, trainable=True)
        self.W1 = keras.layers.Dense(hidden_dimensions, use_bias=False)
        self.W2 = keras.layers.Dense(hidden_dimensions, use_bias=False)
        self.V = keras.layers.Dense(1, use_bias=False)

    def call(self, encoder_outputs, dec_output):
        """
        输入编码器所有的隐状态，enc_output。 [batch, time_steps, features]
        解码器本次解码的状态d，state_h。[batch, features]
        :param encoder_outputs: [batch, time_steps, features]
        :param dec_output: [batch, features]

        :return: 指针的概率分布，一共有time_steps种可能，p [batch,time_steps]

        """
        w1_e = self.W1(encoder_outputs)  # [batch, time_steps, hidden_dimensions]
        w2_d = self.W2(dec_output)  # [batch,hidden_dimensions]
        w2_d = K.expand_dims(w2_d, axis=1)  # [batch,1,hidden_dimensions]
        tanh_output = tanh(w1_e + w2_d)  # [batch,time_steps,hidden_dimensions]
        v_dot_tanh = self.V(tanh_output)  # [batch,time_steps,1]
        attention_weights = softmax(v_dot_tanh, axis=1)  # [batch,time_steps,1]
        att_shape = K.shape(attention_weights)
        p = K.reshape(attention_weights, (att_shape[0], att_shape[1]))
        return p


class Decoder(keras.layers.Layer):
    """
        PointerLSTM
    """

    def __init__(self, hidden_dimensions=128, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.hidden_dimensions = hidden_dimensions
        self.attention = Attention(hidden_dimensions)
        self.decoder_cell = DecoderCell(hidden_dimensions)

    def build(self, input_shape):
        super(Decoder, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, enc_output, states):
        """

        :param enc_output: encoder的输出，enc_output: [h1,h2,h3...h_n] [batch,time_steps,hidden_dimensions]
        :param states: [h,c] 两个状态向量的list
        :return:
        """
        self.enc_output = enc_output
        input_shape = enc_output.shape
        x_input = enc_output[:, enc_output.shape[1] - 1, :]
        x_input = K.repeat(x_input, input_shape[1])
        """
        进入rnn函数，在x_input上，在时间戳维度上进行迭代执行step函数。
        step 函数需要的状态共有3个，分别为[h,c,last_pointer], 第一次的last_pointer为0张量
        """
        b = tf.shape(enc_output)[0]
        # if enc_output.shape[0] is None:
        #     b = 2
        # else:
        #     b = enc_output.shape[0]
        last_pointer = tf.ones(shape=(b, enc_output.shape[1]))
        initial_states = states + [last_pointer]
        last_output, outputs, states = K.rnn(self.step, x_input,
                                             initial_states)

        return outputs

    def step(self, x_input, states):
        """
        对于这个K.rnn，x_input已经定死了，只能是h_n，所以只能通过状态来输入上一次的结果。
        :param x_input: [batch,hidden_dimensions],是编码器最后一次的隐状态, 且每次进来都是
        :param states: [h,c,last_pointer]
        :return:
        """
        h, c, last_pointer = states
        _, [h, c] = self.decoder_cell(last_pointer, [h,c])
        # probs 是 [batch,输入时间戳]大小的张量
        probs = self.attention(self.enc_output, h)
        return probs, [h, c, probs]

    def get_initial_state(self,states):
        """
        根据 states的形状来生成一个初始的 last_pointer 张量
        :param states: [h,c]
        :return:
        """
