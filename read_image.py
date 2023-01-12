"""
@Time : 2023/1/12 13:45
@author:RW，NanJin Ank.
@FileName: read_image.py
@Dis: 利用opencv进行图像读取
@link
"""

# https://blog.csdn.net/youcans/article/details/121168935
# https://blog.csdn.net/youcans/article/details/121169054

import os
import cv2 as cv
import numpy as np
import urllib.request as request
import matplotlib.pyplot as plt
import sys
sys.path.append('E:\Pycode\Opencv_300')

def show_img(img_arr, name):

    # 如果要显示大于屏幕分辨率的图像，需要先调用 namedWindow(“”，WINDOW_NORMAL)
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, img_arr)
    cv.waitKey(0)

# 本地读取
def read_img_from_local(img_path):

    img_1 = cv.imread(img_path, flags=1) # 全彩
    img_2 = cv.imread(img_path, flags=0) # 灰度形式读取

    show_img(img_1, 'color')
    show_img(img_2, "gray")

# 利用网址连接进行读取， 原理：网络上的数据需要读取后进行解码
def read_img_from_net(url_path):
    response = request.urlopen(url_path)
    img= cv.imdecode(np.array(bytearray(response.read()), dtype=np.uint8), -1)
    show_img(img, 'net')


# 保存图像
def imwrite_img(img_arr, save_path):

    cv.imwrite(save_path, img_arr)

# ToDo: 图像的直接拼接，后续需要做
def mosaick_multi_img(img_1, img_2):

    img_stack = np.stack((img_1, img_2))
    cv.imshow('demo', img_stack)
    cv.waitKey(0)

# cv图像在matplotlib上显示
def convert_Mat_To_Matplotlib(img_arr):

    tmp_arr = cv.cvtColor(img_arr, cv.COLOR_BGR2RGB)
    plt.imshow(tmp_arr)
    plt.show()

def select_img_roi(img_arr):

    roi = cv.selectROI(img_arr, showCrosshair=True, fromCenter=False)
    roi_arr = img_arr[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    show_img(roi_arr, 'roi')

# 修改图像像素
def change_imgval_by_index(img_arr, x,y,c=0, change_val=15):

    print('ori_img val: %d'%(img_arr[y,x,c]))
    img_arr[y,x,c] = change_val
    print('now_img val: %d'%(img_arr[y,x,c]))

# 图像通道拆分
def split_img(img_arr):
    if (img_arr.shape[2] ==3):
        bImg, gImg, rImg = cv.split(img_arr)

        cv.imshow('rImg', rImg)
        cv.imshow('bImg', bImg)
        cv.waitKey(0)

# 图像通道合并
def merge_img(bImg, g_Img, r_Img):

    imgMerge = cv.merge([bImg, g_Img, r_Img])
    show_img(imgMerge,'merge')


if __name__ == "__main__":

    img_path = r'E:\Pycode\Opencv_300\Dataset\lena.png'

    if os.path.exists(img_path) == False:
        print('图像不存在')
        exit()

    # img_1, img_2 = read_img_from_local(img_path)

    url_img_path = "https://profile.csdnimg.cn/8/E/F/0_youcans"
    # read_img_from_net(url_img_path)

    img_arr = cv.imread(img_path, flags=1)
    save_img_path = r'E:\Pycode\Opencv_300\Dataset\test/imwrite_lena.png'
    # imwrite_img(img_arr, save_img_path)

    # mosaick_multi_img(img_arr, img_arr)
    # convert_Mat_To_Matplotlib(img_arr)
    # print(img_arr.shape)
    # change_imgval_by_index(img_arr, 472,510,2)
    # select_img_roi(img_arr)
    # split_img(img_arr)
    # bImg, g_Img, r_Img = cv.split(img_arr)
    # merge_img(bImg, g_Img, r_Img)
