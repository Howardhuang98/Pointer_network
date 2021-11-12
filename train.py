#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/10 15:38  
------------      
"""
import time
import numpy as np
from keras.layers import Input
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from model import *

# 记录脚本运行时间
time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# 加载数据
X = np.load(r"tsp_data/tsp_5/tsp5_train_x.npy")
YY = np.load(r"tsp_data/tsp_5/tsp5_train_y.npy")
x_train, x_test, y_train, y_test = train_test_split(X, YY, test_size=0.2)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)
print(x_train.shape, y_train.shape)

# 构建模型
main_input = Input(shape=(X.shape[1], 2), name='main_input')
enc_output, state_h, state_c = Encoder()(main_input)
outputs = Decoder()(enc_output, [state_h, state_c])
model = Model(main_input, outputs)
print(model.summary())
# 指定训练配置
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='./data/ckp-{}/checkpoint'.format(time),
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
early_stop_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=50,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
history = model.fit(x_train,
                    y_train,
                    epochs=1000,
                    validation_data=(x_valid, y_valid),
                    batch_size=128,
                    callbacks=[model_checkpoint_callback, early_stop_callback])
loss, acc = model.evaluate(x_test, y_test)
print("测试集损失：{} \n"
      "测试集准确率：{}".format(loss,acc))
# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
