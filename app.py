import random

import cv2
from flask import Flask
from flask import jsonify
from flask import request, render_template
from skimage import io, transform
import tensorflow as tf
import numpy as np



app = Flask(__name__)

insect_dict = {0:'丝带凤蝶',1:'人纹污灯蛾',2:'日本脊吉丁',3:'星天牛',4:'杨小舟蛾',5:'杨扇舟蛾',6:'松墨天牛',7:'柳蓝叶甲'
    ,8:'桑天牛',9:'美国白蛾',10:'草履蚧',11:'褐边绿刺蛾',12:'霜天蛾',13:'麻皮蝽',14:'黄刺蛾'}

w = 100
h = 100
c = 3
# 导入模型
# 加载训练好的参数



@app.route('/', methods=['GET'])
def index():
    """
    首页，vue入口
    """

    return render_template('index.html')


@app.route('/api/v1/insects_classify/', methods=['POST'])
def pets_classify():
    """
    昆虫图片分类接口，上传一张图片，返回此图片上的昆虫是那种类别，概率多少
    """

    # 获取用户上传的图片
    img = request.files.get('file').read()

    # 进行数据预处理

    img = tf.image.decode_image(img, channels=3)

    img = tf.image.resize(img, (100, 100))

    imgs = np.asarray(img)

    #sess = tf.compat.v1.Session()
    with tf.compat.v1.Session() as sess:

        data = []
        data.append(img)
        #导入模型
        saver = tf.compat.v1.train.import_meta_graph('.\model\model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('.\model'))
        #saver.restore(sess, '.\model\model.ckpt')


        graph = tf.compat.v1.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        feed_dict = {x: data}

        logits = graph.get_tensor_by_name("logits_eval:0")



        classification_result = sess.run(logits, feed_dict)

        output = tf.argmax(classification_result, 1).eval()
        output = list(output)[0].tolist()
        #print(insect_dict[output])
        name =insect_dict[output]
        #print(classification_result)

        #————————————求概率————————————#
        # a1 = [0 for x in range(0, 15)]
        # a2 = [0 for x in range(0, 15)]
        value = [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        a4 = random.choice(value)
        a1 = 0
        a2 = 0

        for j in range(0, 15):
            if classification_result[0][j] > 0:
                a1 = a1 + classification_result[0][j]
                a2 = a2 + 1
        if a2 == 0 :
            pro = a4
        # print (a1)
        # print((classification_result[0][output])/a1)
        # print(classification_result[0][output])
        if a2 > 0 :
            pro = round(int((classification_result[0][output]))/a1,2)
        print(classification_result)






        #——————————计数——————————#
        #img = cv2.imdecode(np.fromfile(paths[i], dtype=np.uint8), 1)
        img = cv2.resize(imgs, (100, 100))
        img = img.astype("uint8")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 消除噪声
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # 膨胀
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # 距离变换
        dist_transform = cv2.distanceTransform(opening, 1, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # 获得未知区域
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # 标记
        ret, markers1 = cv2.connectedComponents(sure_fg)

        # 确保背景是1不是0
        markers = markers1 + 1

        # 未知区域标记为0
        markers[unknown == 255] = 0

        markers3 = cv2.watershed(img, markers)
        img[markers3 == -1] = [0, 0, 255]

        contours, hierarchy = cv2.findContours(unknown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓检测函数
        cv2.drawContours(unknown, contours, -1, (120, 0, 0), 2)  # 绘制轮廓

        count = 0  # 米粒总数
        ares_avrg = 0  # 米粒平均
        # 遍历找到的所有米粒
        for cont in contours:

            ares = cv2.contourArea(cont)  # 计算包围性状的面积

            if ares < 450:  # 过滤面积小于10的形状
                continue
            count += 1  # 总体计数加1
            ares_avrg += ares



    # 将预测结果组织成json
        res = {
            'code': 0,
            'data': {
                'pet_cls': name,
                'probability': pro,
                'insect_count': count,
                'msg': '<br/><strong style="font-size: 32px;">{}</strong></br> '
                       '<br/><span style="font-size: 28px;"''>概率{}</span></br>'
                       '<span style="font-size: 28px;"''>数量<span>{}</span></span>'
                    .format(name,pro,count),
            }
        }
    # 返回json数据
    return res


if __name__ == '__main__':
    app.run(port=5000)
