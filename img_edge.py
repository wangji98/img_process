import cv2
import numpy as np
import matplotlib.pyplot as plt


class Edge:
    def __init__(self,img):
        self.img = img
        self.lenna_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.grayImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.gaussianBlur = cv2.GaussianBlur(self.grayImage, (3,3), 0)
        ret,self.binary = cv2.threshold(self.gaussianBlur, 127, 255, cv2.THRESH_BINARY)
    def Roberts(self):
        #Roberts算子
        kernelx = np.array([[-1,0],[0,1]], dtype=int)
        kernely = np.array([[0,-1],[1,0]], dtype=int)
        x = cv2.filter2D(self.binary, cv2.CV_16S, kernelx)
        y = cv2.filter2D(self.binary, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Roberts
    def Prewitt(self):
        #Prewitt算子
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
        x = cv2.filter2D(self.binary, cv2.CV_16S, kernelx)
        y = cv2.filter2D(self.binary, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return Prewitt
    def Sobel(self):
        #Sobel算子
        x = cv2.Sobel(self.binary, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(self.binary, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Sobel
    def Laplacian(self):
        #拉普拉斯算法
        dst = cv2.Laplacian(self.binary, cv2.CV_16S, ksize = 3)
        Laplacian = cv2.convertScaleAbs(dst)
        return Laplacian
    def Scharr(self):
        # Scharr算子
        x = cv2.Scharr(self.grayImage, cv2.CV_32F, 1, 0)  # X方向
        y = cv2.Scharr(self.grayImage, cv2.CV_32F, 0, 1)  # Y方向
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Scharr = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        return Scharr
    def Canny(self):
        # Canny算子
        gaussian = cv2.GaussianBlur(self.grayImage, (3, 3), 0)  # 高斯滤波降噪
        Canny = cv2.Canny(gaussian, 50, 150)
        return Canny
    def LOG(self):
        # LOG算子
        gaussian = cv2.GaussianBlur(self.grayImage, (3, 3), 0)  # 先通过高斯滤波降噪
        dst = cv2.Laplacian(gaussian, cv2.CV_16S, ksize=3)  # 再通过拉普拉斯算子做边缘检测
        LOG = cv2.convertScaleAbs(dst)
        return LOG

    def plot(self):
        #效果图
        titles = ['Source Image', 'Binary Image', 'Roberts Image',
                  'Prewitt Image','Sobel Image', 'Laplacian Image',
                  'Scharr Image','Canny Image','LOG Image']
        images = [self.lenna_img, self.binary,self.Roberts(), self.Prewitt(), self.Sobel(), self.Laplacian(),
                  self.Scharr(),self.Canny(),self.LOG()]
        for i in np.arange(9):
           plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
           plt.title(titles[i])
           plt.xticks([]),plt.yticks([])
        plt.tight_layout()
        plt.savefig('save_plot/edge.jpg')
        plt.show()
