# -*- coding:utf-8 -*-
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import os

path= "images/"  # 图片所在的文件夹
filename_list = os.listdir(path)
for name in filename_list:
    folder=name
    portion = os.path.splitext(name)  # 将文件名拆成名字和后缀
    savepath=path + folder   #图片保存的文件夹
    # os.mkdir(path)

    #遍历该目录下的所有图片文件
    # for filename in os.listdir(savepath):
    #     uint24_img = cv2.imread(savepath + '/' + filename,0)

#
# path = "./all"
#
# filename_list = os.listdir(path)
    pre_savepath = "time_test_pre/{}".format(folder)  # 图片保存的文件夹
    os.mkdir(pre_savepath)

    for i in os.listdir(savepath):
        file = savepath + '/' + i
        portion = os.path.splitext(i)  # 将文件名拆成名字和后缀


        # 提升对比度
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 线性变换
        a = 2
        im = float(a) * img
        im[im > 255] = 255  # 大于255要截断为255
        # 数据类型的转换
        im = np.round(im)
        im = im.astype(np.uint8)




        # 设置SimpleBlobDetector_Params参数
        params = cv2.SimpleBlobDetector_Params()

        params.minDistBetweenBlobs = 10
        # 合并： 计算二进制图像中二进制斑点的重心，并合并更靠近minDistBetweenBlobs的斑点
        params.filterByInertia = False

        params.filterByConvexity = False
        # 颜色区分,提取暗点不提取亮点
        params.filterByColor = False
        # params.blobColor = 0

        params.filterByCircularity = False

        params.filterByArea = False
        # 改变阈值
        params.minThreshold = 127
        params.maxThreshold = 255
        # 通过面积滤波
        # 这里的面积是基于像素单位的，主要是利于几何矩进行计算得到。
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 5000
        # 通过圆度滤波
        # 当C等于1时候，该形状表示一个完美的圆形
        # 当C趋近于0的时候，该形状表示接近于直线的多边形或者矩形。
        # 当C值在0.75 ~ 0.85之间的时候，多数的时候表示与矩形或者等边的多边形出现。
        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.maxCircularity = 1
        # 通过凸度滤波
        # 表示几何形状是凸包还是凹包的度量。说白了就是可以根据参数过滤凸多边形还是凹多边形， 输入的参数一般在0~1之间，最小为0，最大为1。一般圆形多会大于0.5以上
        params.filterByConvexity = False
        params.minConvexity = 0.9
        # 通过惯性比滤波
        # 惯性率是跟偏心率，圆形的偏心率等于0， 椭圆的偏心率介于0和1之间，直线的偏心率接近于0， 基于几何矩计算惯性率比计算偏心率容易，所以OpenCV选择了惯性率这个特征值，根据惯性率可以计算出来偏心率
        params.filterByInertia = True
        params.minInertiaRatio = 0.001
        # 创建一个检测器并使用默认参数
        time_start = time.process_time()
        detector = cv2.SimpleBlobDetector_create(params)
        # 检测blobs.
        key_points = detector.detect(im)


        if len(key_points) < 1:
            #     # 黑白反转
            #     # img=cv2.imread('219.jpg',1)
            #     # gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img_info = img.shape
            image_height = img_info[0]
            image_weight = img_info[1]
            im = np.zeros((image_height, image_weight, 1), np.uint8)
            for i in range(image_height):
                for j in range(image_weight):
                    imgPixel = img[i][j]
                    im[i][j] = 127 - imgPixel

            key_points = detector.detect(im)
        # 绘制blob的红点
        draw_image = cv2.drawKeypoints(im, key_points, np.array([]), (0, 0, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        time_cost = time.process_time() - time_start
        print('this image time cost: {}'.format(time_cost))



        # 保存预测结果
        cv2.imwrite("./draw_img/{}_pre.jpg".format(portion[0]),draw_image)


        # 统一图片尺寸，可以自定义设置（宽，高）
        img1 = Image.open(file)
        img2 = Image.open("./draw_img/{}_pre.jpg".format(portion[0]))
        # 自定义设置宽高(读者自行修改要拼接的图片分辨率)
        size1, size2 = img1.size
        print(size1,size2)
        joint = Image.new('RGB', (size1 * 2, size2))
        loc1, loc2 = (0, 0), (size1, 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(pre_savepath + "/{}_joint.jpg".format(portion[0]))




        # plt.imshow(joint)
        # plt.show()


        # Show blobs
        # plt.imshow( draw_image)
        # plt.show()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        print("ROI区域包含细胞数:"+str(len(key_points)))


