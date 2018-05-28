# -*- coding:UTF-8 -*-
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import graph_util
import tensorflow as tf
import inference
import os
import numpy as np


BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0005
TRAINING_STEPS = 60000
MOVING_AVERAGE_DECAY = 0.99
DROPOUT = 0.8

NUM_CHANNELS = 3
IMAGE_SIZE = 227
OUTPUT_NODE = 2

MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'model.ckpt'

path = "output.tfrecords"
files = tf.train.match_filenames_once(path)#获取所有符合正则表达式的文件,返回文件列表
filename_queue = tf.train.string_input_producer([path],shuffle=False)  # create a queue

def train():
    #准备读数据
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file_name and file

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'width': tf.FixedLenFeature([], tf.int64),
                                           'height': tf.FixedLenFeature([], tf.int64),
                                           'channels': tf.FixedLenFeature([], tf.int64)
                                       })  # return image and label

    # img = test_tf.image.convert_image_dtype(img, dtype=test_tf.float32)
    # img = test_tf.reshape(img, [512, 80, 3])  # reshape image to 512*80*3
    # img = test_tf.cast(img, test_tf.float32) * (1. / 255) - 0.5  # throw img tensor

    label = tf.cast(features['label'], tf.int32)  # throw label tensor
    # height = tf.cast(features['height'],tf.int32)
    height = features['height']
    width = tf.cast(features['width'], tf.int32)
    channels = tf.cast(features['channels'], tf.int32)

    img = tf.decode_raw(features['img'], tf.uint8)
    print(type(height))
    img = tf.reshape(img, [227, 227, 3])

    # img.set_shape([324,324,1])
    # label = test_tf.reshape(label,[1])
    # img = test_tf.image.resize_images(img,[width,height],method=1)

    # label = test_tf.reshape(label,[1])
    # img.set_shape([height,width,1])
    # img.set_shape([height,width])
    # img = test_tf.image.convert_image_dtype(img,dtype=test_tf.float32)
    # img = test_tf.image.resize_images(img,[height,width],method=0)

    # img=test_tf.cast(img,test_tf.float32)*(1./255)-0.5

    #
    # batch_size = 64
    # min_after_dequeue = 10
    # capacity = 100 + 3 * batch_size
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=50,
                                                    capacity=22000,
                                                    min_after_dequeue=20000,
                                                    num_threads=4)
    #
    img_batch1, label_batch1 = tf.train.shuffle_batch([img, label],
                                                                batch_size=1000,
                                                                capacity=22000,
                                                                min_after_dequeue=20000,
                                                                num_threads=4)

    #定义预输入
    # label_batch = tf.reshape(label_batch, [64])
    x = tf.placeholder(tf.float32,
                       [None,
                       IMAGE_SIZE,
                       IMAGE_SIZE,
                       NUM_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.int64,
                        [None],
                        name='y-input')

    #定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

    #调用神经网计算输出结果
    y = inference.alex_net(X=x,output=OUTPUT_NODE,dropout=DROPOUT,regularizer=regularizer)
    result = tf.argmax(y,1,name='out')

    #定义统计训练轮数的全局变量
    global_step = tf.Variable(0,trainable=False)
    #定义滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #准确率
    correct_prediction = tf.equal(tf.argmax(y, 1),y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
   # loss = cross_entropy_mean
    #定义学习率变化
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
       200,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    # train_step = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)
    #同时更新滑动平均和网络参数
    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # sess.run(test_tf.local_variables_initializer())#使用tf.train.match_filenames_once(path)需要这句
        sess.run(tf.global_variables_initializer())




        for i in range(TRAINING_STEPS):


            xs, ys = sess.run([img_batch, label_batch])

           
            _,loss_value,step= sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})

            if i%100 == 0:
                print('After %d training steps,loss on training batch is %g'%(step,loss_value))
                #保存checkpoint文件
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
                #保存pb文件
                output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])
                with tf.gfile.GFile('model_pb/combined_model.pb', 'wb') as f:
                    f.write(output_graph_def.SerializeToString())
            if i%1000 == 0:
                xs1, ys1 = sess.run([img_batch1, label_batch1])
                
                ac = sess.run(accuracy,feed_dict={x:xs1,y_:ys1})
                print('accuracy(1000):%s'%ac)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    train()

