# -*- coding:UTF-8 -*-
import tensorflow as tf
import numpy as np

#定义卷积操作函数
def conv2d(name,x,weights,bias):
    con = tf.nn.conv2d(x,weights,strides=[1,1,1,1],padding='SAME')
    rel = tf.nn.relu(tf.nn.bias_add(con,bias),name=name)
    return rel
#定义池化操作函数
def max_pool(name,x,k):
    return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,2,2,1],padding='SAME',name=name)
#定义归一化操作函数
def norm(name,x,lsize=4):
    return tf.nn.lrn(x,lsize,bias=1.0,alpha=0.0001,beta=0.75,name=name)


#定义Alex网络
def alex_net(X,output,dropout,regularizer):
    Weights = {
        # 'wc1':tf.get_variable([3,3,1,64],initializer=tf.truncated_normal_initializer(stddev=0.1)),
        'wc1': tf.Variable(tf.truncated_normal([7, 7, 3, 96], stddev=0.01)),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.01)),
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.01)),
        'wf1': tf.Variable(tf.truncated_normal([6*6*384, 512], stddev=0.005)),
        'wf2': tf.Variable(tf.truncated_normal([512,512], stddev=0.005)),
        'wo': tf.Variable(tf.truncated_normal([512, 2], stddev=0.01))
    }

    biases = {
        'bc1': tf.Variable(tf.zeros([96])),
        'bc2': tf.Variable(tf.zeros([256])),
        'bc3': tf.Variable(tf.zeros([384])),
        'bf1': tf.Variable(tf.zeros([512])),
        'bf2': tf.Variable(tf.zeros([512])),
        'fo': tf.Variable(tf.zeros([output]))
    }

    #构造第一个卷积层

    conv1 = tf.nn.conv2d(X, Weights['wc1'], strides=[1, 4, 4, 1], padding='VALID')
    rel1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['bc1']), name='conv1')
  #  pool1 = max_pool('pool1',rel1,k=3)
    pool1 = tf.nn.max_pool(rel1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
    norm1 = norm('norm1',pool1,lsize=5)
    if regularizer!=None:
       tf.add_to_collection('losses', regularizer(Weights['wc1']))
    #drop1 = tf.nn.dropout(norm1,dropout)

    #构造第二个卷积层

#    inp = tf.pad(norm1,paddings=[[2,0],[2,0]],mode = 'CONSTANT',name=None,constant_values=0)
    conv2 = tf.nn.conv2d(norm1,Weights['wc2'],strides=[1,1,1,1],padding='SAME')
    rel2 = tf.nn.relu(tf.nn.bias_add(conv2,biases['bc2']),name='conv2')
    pool2 = tf.nn.max_pool(rel2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
    norm2 = norm('norm2',pool2,lsize=5)
    if regularizer!=None:
        tf.add_to_collection('losses', regularizer(Weights['wc2']))
   # drop2 = tf.nn.dropout(norm2,dropout)

    #够造第三个卷积层
    conv3 = tf.nn.conv2d(norm2,Weights['wc3'],strides=[1,1,1,1],padding='SAME')
    rel3 = tf.nn.relu(tf.nn.bias_add(conv3,biases['bc3']),name='conv3')
   # pool3 = max_pool('pool3',rel3,k=3)
    pool3 =  tf.nn.max_pool(rel3,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool3')
  #  norm3 = norm('norm3',pool3,lsize=4)
    if regularizer!=None:
        tf.add_to_collection('losses', regularizer(Weights['wc3']))
  #  drop3 = tf.nn.dropout(norm3,dropout)

    reshaped = tf.reshape(pool3,[-1,Weights['wf1'].get_shape().as_list()[0]])


    #构造第一个全连接层
    #weit = tf.get_variable('weight',[nodes,512],initializer=tf.truncated_normal_initializer(stddev=0.1))
    #bia = tf.get_variable('bias',[512],initializer=tf.constant_initializer(0.1))
    fc1 = tf.nn.relu(tf.matmul(reshaped,Weights['wf1'])+biases['bf1'],name='fc1')
    fc1 = tf.nn.dropout(fc1,dropout)
    if regularizer!=None:
        tf.add_to_collection('losses', regularizer(Weights['wf1']))
    #构造第二个全连接层
    fc2 = tf.nn.relu(tf.matmul(fc1,Weights['wf2'])+biases['bf2'],name='fc2')
    fc2 = tf.nn.dropout(fc2,dropout)
    if regularizer!=None:
        tf.add_to_collection('losses', regularizer(Weights['wf2']))
    #构造输出层
    result = tf.matmul(fc2,Weights['wo'])+biases['fo']
    if regularizer!=None:
        tf.add_to_collection('losses', regularizer(Weights['wo']))

    return result


