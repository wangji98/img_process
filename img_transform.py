
import cv2
import numpy as np
import matplotlib.pyplot as plt

class IamgeTransform:

    def __init__(self,img):
        self.img = img

    def ImgResize(self):
        src = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        rows, cols = src.shape[:2]
        print(rows, cols)
        # 图像缩放
        result = cv2.resize(src, None, fx=0.3, fy=0.3)
        result1 = cv2.resize(src, None, fx=1.3, fy=1.3)
        # 显示图形
        titles = ['Source', 'Image1', 'Image2']
        images = [src, result, result1]
        for i in range(3):
            plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def ImgRotation(self):
        # 原图的高、宽 以及通道数
        rows, cols, channel = self.img.shape

        # 绕图像的中心旋转
        # 参数：旋转中心 旋转度数 scale
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
        # 参数：原始图像 旋转参数 元素图像宽高
        rotated = cv2.warpAffine(self.img, M, (cols, rows))

        # 显示图像
        cv2.imshow("src", self.img)
        cv2.imshow("rotated", rotated)

        # 等待显示
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def ImgFlip(self):
        src = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 图像翻转
        # 0以X轴为对称轴翻转 >0以Y轴为对称轴翻转 <0X轴Y轴翻转
        img1 = cv2.flip(src, 0)
        img2 = cv2.flip(src, 1)
        img3 = cv2.flip(src, -1)

        # 显示图形
        titles = ['Source', 'Image1', 'Image2', 'Image3']
        images = [src, img1, img2, img3]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

    def ImgTranslata(self):
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 图像平移 下、上、右、左平移
        M = np.float32([[1, 0, 0], [0, 1, 100]])
        img1 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        M = np.float32([[1, 0, 0], [0, 1, -100]])
        img2 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        M = np.float32([[1, 0, 100], [0, 1, 0]])
        img3 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        M = np.float32([[1, 0, -100], [0, 1, 0]])
        img4 = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # 显示图形
        titles = ['Image1', 'Image2', 'Image3', 'Image4']
        images = [img1, img2, img3, img4]
        for i in range(4):
            plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

