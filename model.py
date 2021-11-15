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

    def __init__(self, hidden_dimensions=128, name="attention"):
        super(Attention, self).__init__(name=name, trainable=True)
        self.W1 = keras.layers.Dense(hidden_dimensions, use_bias=False, input_shape=(2,))
        self.W2 = keras.layers.Dense(hidden_dimensions, use_bias=False, input_shape=(2,))
        self.V = keras.layers.Dense(1, use_bias=False,input_shape=(hidden_dimensions,))

    def call(self, encoder_outputs, dec_output):
        """
        输入编码器所有的隐状态，enc_output。 [batch, time_steps, features]
        解码器本次解码的状态d，state_h。[batch, features]
        :param encoder_outputs: [batch, time_steps, features]
        :param dec_output: [batch, features]

        :return: 指针的概率分布，一共有time_steps种可能，p [batch,time_steps]

        """
        # w1_e = self.W1(encoder_outputs)  # [batch, time_steps, hidden_dimensions]
        # w2_d = self.W2(dec_output)  # [batch,hidden_dimensions]
        # w2_d = K.expand_dims(w2_d, axis=1)  # [batch,1,hidden_dimensions]
        # tanh_output = tanh(w1_e + w2_d)  # [batch,time_steps,hidden_dimensions]
        # v_dot_tanh = self.V(tanh_output)  # [batch,time_steps,1]
        # attention_weights = softmax(v_dot_tanh, axis=1)  # [batch,time_steps,1]
        # att_shape = K.shape(attention_weights)
        # p = K.reshape(attention_weights, (att_shape[0], att_shape[1]))
        # return p
        # 此处outputs 是一个 [batch,time_steps,1]的张量
        batch = tf.shape(encoder_outputs)[0]
        time_steps = tf.shape(encoder_outputs)[1]
        _, outputs, _ = K.rnn(self.step,encoder_outputs,initial_states=[dec_output])
        outputs = tf.reshape(outputs,shape=[batch,time_steps])
        return outputs

    def step(self,input,states):
        """

        :param input: [batch, features]
        :param states: dec_output [batch, hidden_dimensions]
        :return:
        """
        dec_output = states[0]
        w1_e = self.W1(input)
        w2_d = self.W2(dec_output)
        tanh_output = tanh(w1_e + w2_d)
        v_dot_tanh = self.V(tanh_output)
        return v_dot_tanh, [dec_output]


class Decoder(keras.layers.Layer):
    """
        PointerLSTM
    """

    def __init__(self, hidden_dimensions=128, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.hidden_dimensions = hidden_dimensions
        self.attention = Attention(hidden_dimensions)
        self.decoder_cell = DecoderCell(hidden_dimensions)
        self.x = None

    def build(self, input_shape):
        super(Decoder, self).build(input_shape)
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x, enc_output, states):
        """

        :param x: [batch, time_steps, features]
        :param enc_output: encoder的输出，enc_output: [h1,h2,h3...h_n] [batch,time_steps,hidden_dimensions]
        :param states: [h,c] 两个状态向量的list
        :return:
        """
        """
        进入rnn函数，在x_input上，在时间戳维度上进行迭代执行step函数。
        step 函数需要的状态共有3个，分别为[h,c,last_pointer], 第一次的last_pointer为0张量。
        """
        self.x = x
        b = tf.shape(enc_output)[0]
        last_pointer = tf.ones(shape=(b, 2))
        initial_states = states + [last_pointer, enc_output, x]
        last_output, outputs, states = K.rnn(self.step, enc_output,
                                             initial_states)

        return outputs

    def step(self, x_input, states):
        """
        对于这个K.rnn，x_input已经定死了，只能是h_n，所以只能通过状态来输入上一次的结果。
        :param x_input: [batch,hidden_dimensions],是编码器最后一次的隐状态, 且每次进来都是
        :param states: [h,c,last_pointer,enc_output]
        :return:
        """
        h, c, last_pointer, enc_output, x = states
        _, [h, c] = self.decoder_cell(last_pointer, [h, c])
        # probs 是 [batch,输入时间戳]大小的张量
        probs = self.attention(enc_output, h)
        pointer = _get_pointer(x, probs)
        return probs, [h, c, pointer, enc_output, x]


def _get_pointer(x, probs):
    """
    :param x: [batch,time_steps,2]
    :param probs: [batch,time_steps]
    :return: [batch,2]
    """
    idx = tf.argmax(probs,axis=1)
    pointer = tf.gather(x,idx,batch_dims=1)
    return pointer

