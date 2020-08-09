#!/usr/bin/env python
# encoding: utf-8
'''
@author: songyunlong
@license: (C) Copyright 2020-2025.
@contact: 1243049371@qq.com
@software: pycharm
@file: model_structure
@time: 2020/8/7 5:17 下午
'''

import tensorflow as tf
import numpy as np

class InteractionLayer(tf.keras.layers.Layer):
    def __init__(self, latent_num:int):
        '''
        init
        :param latent_num: 嵌入层维度
        '''
        super().__init__(name='interaction')
        self._latent_num = latent_num
    def build(self, input_shape):
        # print(input_shape)
        self.embedding = self.add_weight(name='embedding', shape=[input_shape[-1], self._latent_num],
                                         dtype=tf.float32, trainable=True,
                                         initializer=tf.keras.initializers.TruncatedNormal(mean=0, stddev=0.01))
    def call(self, inputs, **kwargs):
        '''
        call
        :param input: 输入数据, 维度为[None, input_shape]
        :return:
        '''
        interaction_output = tf.constant(value=0, dtype=tf.float32)
        for f in range(self.embedding.shape[-1]):
            ex2 = tf.square(tf.reduce_sum(self.embedding[:, f] * inputs))
            e2x = tf.reduce_sum(tf.square(self.embedding[:, f] * inputs))
            interaction_output += (ex2 - e2x)
        return interaction_output * 0.5

class Modelfm(tf.keras.Model):
    def __init__(self, latent_num:int):
        '''
        init
        :param latent_num: 嵌入层维度
        '''
        super().__init__()
        self._linear_part = tf.keras.layers.Dense(units=1, use_bias=True, name='linear')
        self._interaction_part = InteractionLayer(latent_num=latent_num)

    def call(self, inputs, training=None, mask=None):
        # print(inputs.shape)
        linear = self._linear_part(inputs)
        interaction = self._interaction_part(inputs)
        return tf.keras.activations.sigmoid(x=(linear + interaction))

def model_traditionalfit(latent_num:int, train_x, train_y, test_x, test_y):
    '''
    :return:
    '''
    modelfm = Modelfm(latent_num=latent_num)
    modelfm.compile(optimizer=tf.keras.optimizers.Ftrl(), loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy'])
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    modelfm.fit(x=train_x, y=train_y, batch_size=10, epochs=100, verbose=1, shuffle=True, steps_per_epoch=10)
    modelfm.evaluate(test_x, test_y, batch_size=10)
    result = modelfm.predict(test_x, batch_size=10)
    print(result)

def non_keras_training(latent_num:int, train_x, train_y, test_x=None, test_y=None):
    '''
    :return:
    '''
    loss_func = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Ftrl()
    def loss_calc(model:tf.keras.Model, x:tf.Tensor, y:tf.Tensor):
        '''
        损失函数计算
        :param loss_func: 损失函数
        :return:
        '''
        # print(x.shape)
        y_pred = model(inputs=x)
        loss_value = loss_func(y_pred=y_pred, y_true=y)
        return loss_value

    def grad(model:tf.keras.Model, x:tf.Tensor, y:tf.Tensor):
        '''
        梯度计算
        :param model:
        :param x:
        :param y:
        :return:
        '''
        with tf.GradientTape() as tape:
            loss_value = loss_calc(model=model, x=x, y=y)
        return loss_value, tape.gradient(target=loss_value, sources=model.trainable_variables)

    train_loss_results = []
    train_accuracy_result = []
    epoch_num = 100
    modelfm = Modelfm(latent_num=latent_num)

    for epoch in range(epoch_num):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(10).shuffle(True)

        for x, y in dataset:
            # x = x.reshape(1, -1)
            loss_value, grads = grad(model=modelfm, x=x, y=y)
            optimizer.apply_gradients(grads_and_vars=zip(grads, modelfm.trainable_variables))

            epoch_loss_avg(loss_value)
            # print('fff', modelfm(x).shape)
            epoch_accuracy(y, modelfm(x))

        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_result.append(epoch_accuracy.result())

        if epoch % 5 == 0:
            print('Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}'.format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))


if __name__ == '__main__':
    # modelfm = Modelfm(10)
    # # input = tf.keras.Input(shape=(20, ), name='input')
    # output = modelfm(np.array([i for i in range(20)], dtype=np.float32).reshape(1, -1))
    # print(output)
    # # print(modelfm.trainable_variables)
    # for i in modelfm.trainable_weights:
    #     print(i.name, i.shape)
    # rng = np.random.RandomState(0)
    # train_x = np.arange(2000).reshape(100, 20).astype(np.float32)
    # train_y = rng.randint(low=0, high=1, size=(100, )).astype(np.float32)
    # data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(20).shuffle(True)
    # print(data.__iter__().__next__()[-1].shape)

    # test_x = np.arange(200).reshape(10, 20).astype(np.float32)
    # test_y = rng.randint(low=0, high=1, size=(10,)).astype(np.float32)
    # # model_traditionalfit(latent_num=10, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    # non_keras_training(latent_num=10, train_x=train_x, train_y=train_y)

    a = tf.constant(2.)
    b = tf.constant(3.)
    with tf.GradientTape() as tape:
        tape.watch((a, b))
        y = tf.square(a) + tf.square(b)
    i, j = tape.gradient(target=y, sources=(a, b))
    print(i.numpy(), j.numpy())

