# -*- coding:UTF-8 -*-
import time
import tensorflow as tf
import inference
import train_net
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def use(src):


        x = tf.placeholder(tf.float32,
                           [None,
                            train_net.IMAGE_SIZE,
                            train_net.IMAGE_SIZE,
                            train_net.NUM_CHANNELS],
                           name='x-input')
        img = Image.open(src)#输入图片路径
        img_rgb = img.convert("RGB")
        img_rgb = img.resize((227, 227))

        data = img_rgb.getdata()
        data = np.matrix(data, dtype='float')

        xs = tf.reshape(data,[1,227,227,3])

        y= inference.alex_net(X=x,output=2,dropout=train_net.DROPOUT,regularizer=None)
        #tf.argmax()返回向量中最大值位置,tf.equal()返回两个向量对应位置比较结果 返回值为布尔类型


        rel = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(train_net.MOVING_AVERAGE_DECAY)
        #加载变量的滑动平均值
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        #
        # saver = tf.train.Saver()
        cls = 'Male'
        with tf.Session() as sess:
                #返回模型变量取值的路径
                tf.global_variables_initializer().run()

                ckpt = tf.train.get_checkpoint_state(train_net.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #ckpt.model_checkpoint_path返回最新的模型变量取值的路径
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    xs = sess.run(xs)
                    rel = sess.run(rel, feed_dict={x:xs})
                    if(rel==1):
                        cls = 'Female'
                    print('result is %s' % (cls))

                    plt.imshow(img)  # 显示图片
                    plt.axis('on')  # 不显示坐标轴
                    plt.title('%s' % cls)
                    plt.show()
                else:
                    print('NO checkpoint file found')
                    return



if __name__ == '__main__':
    use("D:/1 (1).jpeg")#输入图片url