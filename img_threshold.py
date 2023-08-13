import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageThreshod:
    def __init__(self,img):
        self.img = img
    def Threshold(self):
        # 灰度图像处理

        GrayImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # 阈值化处理
        ret, thresh1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_TOZERO_INV)

        t2, otsu = cv2.threshold(GrayImage, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 显示结果
        titles = ['Gray Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV','BINARY+OTSU']
        images = [GrayImage, thresh1, thresh2, thresh3, thresh4, thresh5,otsu]
        for i in range(6):
            plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.show()
    def AdaptiveThreshold(self):

        GrayImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        ret, th1 = cv2.threshold(GrayImage, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        titles = ['original image', 'global thresholding (v=127)', 'Adaptive mean thres holding',
                  'adaptive gaussian thresholding']
        images = [GrayImage, th1, th2, th3]

        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
        plt.tight_layout()
        plt.show()
