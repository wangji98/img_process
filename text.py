import cv2
import numpy as np
import matplotlib.pyplot as plt
from img_blur import blur
from img_transform import IamgeTransform
def ImgAttribute(img):
    '''

    :param img:
    :return:
    '''
    rows, cols, channel = img.shape
    print('图像形状:', img.shape)
    print('像素数目:', img.size)
    print('图像类型:', img.dtype)
    return rows, cols, channel

def ImgShow(img):
    '''
    :param img:
    :return:
    '''
    cv2.imshow('demo', img)
    # 等待显示
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ImgRead(img_path):
    '''
    :param img_path:
    :return:
    '''
    img = cv2.imread(img_path)
    return img

def BGRChannel(img):
    # 拆分通道
    b,g,r = cv2.split(img)
    # 显示原始图像
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)
    # 合并通道
    m = cv2.merge([b, g, r])
    cv2.imshow("Merge", m)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return b,g,r,m

if __name__ == '__main__':

    img_path = 'Lena.png'
    img = ImgRead(img_path)
    rows, cols, channel = ImgAttribute(img)
    # ImgShow(img)
    # b,g,r,m = BGRChannel(img)
    # # 只显示蓝色通道
    # g = np.zeros((rows,cols),dtype=img.dtype)
    # r = np.zeros((rows,cols),dtype=img.dtype)
    # m = cv2.merge([b,g,r])
    # cv2.imshow('merge',m)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # ImgShow(img)
    # blur(img)
    # ImgRotation(img)
    # ImgFlip(img)
    # ImgTranslata(img)
    # ImgResize(img)
    trans = IamgeTransform(img)
    trans.ImgRotation()
    trans.ImgFlip()
    trans.ImgResize()
    trans.ImgTranslata()