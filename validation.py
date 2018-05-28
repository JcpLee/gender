# -*- coding:UTF-8 -*-
import time
import tensorflow as tf
import inference
import train_net
import numpy as np

EVAL_INTERVAL_SECS = 60

path = "output_validation.tfrecords"
files = tf.train.match_filenames_once(path)#获取所有符合正则表达式的文件,返回文件列表
filename_queue = tf.train.string_input_producer([path],shuffle=False)  # create a queue

def evaluate():
        # with tf.Graph().as_default() as g:
            # x = tf.placeholder(tf.float32, [None, mnist_inferenceLeNet.INPUT_NODE], name='x-input')
        # 准备读数据
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

        label = tf.cast(features['label'], tf.float32)  # throw label tensor
        # height = tf.cast(features['height'],tf.int32)
        height = features['height']
        width = tf.cast(features['width'], tf.int32)
        channels = tf.cast(features['channels'], tf.int32)

        img = tf.decode_raw(features['img'], tf.uint8)
        print(type(height))
        img = tf.reshape(img, [227, 227, 3])



        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size=1000,
                                                        capacity=5819,
                                                        min_after_dequeue=5818,
                                                        num_threads=4)

        x = tf.placeholder(tf.float32,
                           [None,
                            train_net.IMAGE_SIZE,
                            train_net.IMAGE_SIZE,
                            train_net.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(tf.int64, [None], name='y-input')



        # validata_feed = {x: reshape_xs, y_: mnist.validation.labels}

        y= inference.alex_net(X=x,output=2,dropout=train_net.DROPOUT,regularizer=None)
        #tf.argmax()返回向量中最大值位置,tf.equal()返回两个向量对应位置比较结果 返回值为布尔类型


        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
        #数据类型转换
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # variable_averages = tf.train.ExponentialMovingAverage(train_net.MOVING_AVERAGE_DECAY)
        # #加载变量的滑动平均值
        # saver = tf.train.Saver(variable_averages.variables_to_restore())

        #加载保存模型的变量

        saver = tf.train.Saver()

        while True:
            with tf.Session() as sess:
                #返回模型变量取值的路径
                tf.global_variables_initializer().run()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                xs, ys = sess.run([img_batch, label_batch])
                ckpt = tf.train.get_checkpoint_state(train_net.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    #ckpt.model_checkpoint_path返回最新的模型变量取值的路径
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    print('After %s traing steps validation accuracy is %g' % (global_step, sess.run(accuracy, feed_dict={x:xs,y_:ys})))
                else:
                    print('NO checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


if __name__ == '__main__':
    evaluate()