# -*- coding:utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance


# Read image
# im = cv2.imread("230.jpg", cv2.IMREAD_GRAYSCALE)
# 二值化
# img = cv2.imread("215.jpg", cv2.IMREAD_GRAYSCALE)
# ret,im=cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 提升对比度
img = cv2.imread("images/cell_320_0016.JPG", cv2.IMREAD_GRAYSCALE)
# 线性变换
a = 2
im = float(a) * img
im[im > 255] = 255  # 大于255要截断为255
# 数据类型的转换
im = np.round(im)
im = im.astype(np.uint8)



# 设置SimpleBlobDetector_Params参数
params = cv2.SimpleBlobDetector_Params()
# 合并： 计算二进制图像中二进制斑点的重心，并合并更靠近minDistBetweenBlobs的斑点
params.minDistBetweenBlobs = 10

params.filterByInertia = False
params.filterByConvexity = False
# 颜色区分,提取暗点不提取亮点
params.filterByColor = True
params.blobColor = 0

params.filterByCircularity = False

params.filterByArea = False
# 改变阈值
params.minThreshold = 127
params.maxThreshold = 255
# 通过面积滤波
# 这里的面积是基于像素单位的，主要是利于几何矩进行计算得到。
params.filterByArea = True
params.minArea = 25
params.maxArea = 500
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
detector = cv2.SimpleBlobDetector_create(params)
# 检测blobs.
key_points = detector.detect(im)

# 输出各圆心坐标
pts = cv2.KeyPoint_convert(key_points)
print(pts)

# 绘制blob的红点
draw_image = cv2.drawKeypoints(im, key_points, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# Show blobs
plt.imshow(draw_image)
plt.show()

plt.imshow(img, cmap='gray')
plt.show()
print("ROI区域包含细胞数:"+str(len(key_points)))