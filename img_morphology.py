import cv2
import numpy as np
import matplotlib.pyplot as plt
def ImageMorphology(img):

    # 设置卷积核
    kernel = np.ones((5, 5), np.uint8)
    # 图像腐蚀处理
    erosion = cv2.erode(img, kernel)
    # 图像膨胀处理
    dilation = cv2.dilate(img, kernel)
    # 图像开运算
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # 图像闭运算
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # 形态学梯度是通过对原始图像进行膨胀操作，然后对腐蚀操作的结果进行膨胀操作，并将两个结果相减得到的。
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    # 图像顶帽运算
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # 图像黑帽运算
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # 显示结果
    titles = ['Image', 'erosion', 'dilation', 'open', 'close', 'gradient', 'tophat','blackhat']
    images = [img, erosion, dilation, open, close, gradient, tophat,blackhat]
    for i in range(8):
        plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()