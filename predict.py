#!/usr/bin/env python 
# -*- coding:utf-8 -*-
from skimage import io, transform
import tensorflow as tf
import numpy as np
tf = tf.compat.v1

path1 = ".\insecttest/草履蚧/709.jpg"
path2 = ".\insecttest/褐边绿刺蛾/成虫.jpg"
path3 = ".\insecttest/黄刺蛾/成虫.jpg"
path4 = ".\insecttest/柳蓝叶甲/666 (10).jpg"
path5 = ".\insecttest/麻皮蝽/667麻皮蝽.jpg"
path6 = ".\insecttest/美国白蛾/888 (33).jpg"
path7 = ".\insecttest/人纹污灯蛾/999 (43).jpg"
path8 = ".\insecttest/日本脊吉丁/580日本脊吉丁.jpg"
path9 = ".\insecttest/桑天牛/499.jpg"
path10 = ".\insecttest/霜天蛾/104 (19).jpg"
path11 = ".\insecttest/丝带凤蝶/009丝带凤蝶.jpg"
path12 = ".\insecttest/松墨天牛/222 (27).jpg"
path13 = ".\insecttest/星天牛/1876-1F42Q619393M.jpg"
path14 = ".\insecttest/杨扇舟蛾/426.jpg"
path15 = ".\insecttest/杨小舟蛾/426杨小舟蛾.jpg"


flower_dict = {0:'丝带凤蝶',1:'人纹污灯蛾',2:'日本脊吉丁',3:'星天牛',4:'杨小舟蛾',5:'杨扇舟蛾',6:'松墨天牛',7:'柳蓝叶甲'
    ,8:'桑天牛',9:'美国白蛾',10:'草履蚧',11:'褐边绿刺蛾',12:'霜天蛾',13:'麻皮蝽',14:'黄刺蛾'}

w = 100
h = 100
c = 3


def read_one_image(path):
    img = io.imread(path)
    try:
        if img.shape[2] == 3:
            img = transform.resize(img, (w, h))
    except:
        return
    return np.asarray(img)


with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data6 = read_one_image(path6)
    data7 = read_one_image(path7)
    data8 = read_one_image(path8)
    data9 = read_one_image(path9)
    data10 = read_one_image(path10)
    data11 = read_one_image(path11)
    data12 = read_one_image(path12)
    data13 = read_one_image(path13)
    data14 = read_one_image(path14)
    data15 = read_one_image(path15)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)
    data.append(data6)
    data.append(data7)
    data.append(data8)
    data.append(data9)
    data.append(data10)
    data.append(data11)
    data.append(data12)
    data.append(data13)
    data.append(data14)
    data.append(data15)

    saver = tf.train.import_meta_graph('.\model\model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('.\model'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分类
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第", i + 1, "朵花预测:" + flower_dict[output[i]])
        print(output[i])

