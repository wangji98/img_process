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
    def plot(self):
        #效果图
        titles = ['Source Image', 'Binary Image', 'Roberts Image',
                  'Prewitt Image','Sobel Image', 'Laplacian Image']
        images = [self.lenna_img, self.binary,self.Roberts(), self.Prewitt(), self.Sobel(), self.Laplacian()]
        for i in np.arange(6):
           plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
           plt.title(titles[i])
           plt.xticks([]),plt.yticks([])
        plt.show()
