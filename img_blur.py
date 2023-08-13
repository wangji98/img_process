import cv2
import numpy as np
import matplotlib.pyplot as plt


def AddBlur(img):
    rows, cols, chn = img.shape

    # 在图像随机位置加白色噪点
    for i in range(5000):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        img[x, y, :] = 255
    return img
def MeanFilter(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 均值滤波
    result = cv2.blur(source, (5, 5))

    # 显示图形
    titles = ['Source Image', 'MeanFilter Blur Image']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return result

def BoxFilter(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 方框滤波
    result = cv2.boxFilter(source, -1, (5, 5), normalize=1)

    # 显示图形
    titles = ['Source Image', 'BoxFilter Image']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return result

def GaussianBlur(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 高斯滤波
    result = cv2.GaussianBlur(source, (3, 3), 0)

    # 显示图形
    titles = ['Source Image', 'GaussianBlur Image']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return result

def MedianBlur(img):
    source = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 高斯滤波
    result = cv2.medianBlur(source, 3)

    # 显示图形
    titles = ['Source Image', 'MedianBlur Image']
    images = [source, result]
    for i in range(2):
        plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    return result

def blur(img):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgcopy = img.copy()
    source = AddBlur(imgcopy)

    # 均值滤波
    meanresult = cv2.blur(source, (5, 5))
    # 方框滤波
    boxresult = cv2.boxFilter(source, -1, (5, 5), normalize=1)
    # 高斯滤波
    gaussianresult = cv2.GaussianBlur(source, (3, 3), 0)
    # 中值滤波
    medianresult = cv2.medianBlur(source, 3)
    # 显示图形
    titles = ['Source Image','Noise Image', 'MeanBlur Image','BoxFilter Image','GaussianBlur Image','MedianBlur Image']
    images = [img, source, meanresult,boxresult,gaussianresult,medianresult]

    for i in range(6):
        plt.subplot(1, 6, i + 1), plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    return 0