import cv2
import numpy as np
import matplotlib.pyplot as plt
def gary_process(img):
    '''
    :param img:
    :return:
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]

    #创建一幅图像
    maxgrayimg = np.zeros((height, width, 3), np.uint8)
    avegrayimg = np.zeros((height, width, 3), np.uint8)
    weightgrayimg = np.zeros((height, width, 3), np.uint8)
    #图像最大值灰度处理
    for i in range(height):
        for j in range(width):
            #获取图像R G B最大值
            gray = max(img[i,j][0], img[i,j][1], img[i,j][2])
            #灰度图像素赋值 gray=max(R,G,B)
            maxgrayimg[i,j] = np.uint8(gray)
    # 图像平均灰度处理方法
    for i in range(height):
        for j in range(width):
            # 灰度值为RGB三个分量的平均值
            gray = (int(img[i, j][0]) + int(img[i, j][1]) + int(img[i, j][2])) / 3
            avegrayimg[i, j] = np.uint8(gray)
    # 图像平均灰度处理方法
    for i in range(height):
        for j in range(width):
            # 灰度加权平均法
            gray = 0.30 * img[i, j][0] + 0.59 * img[i, j][1] + 0.11 * img[i, j][2]
            weightgrayimg[i, j] = np.uint8(gray)

    # 调用matplotlib显示处理结果
    titles = ['image', 'maxgrayimg', 'avegrayimg', 'weightgrayimg']
    images = [img, maxgrayimg, avegrayimg, weightgrayimg]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()