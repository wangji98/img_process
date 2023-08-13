import cv2
import numpy as np
import matplotlib.pyplot as plt

# 对数变换
def log(c, img):
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output
#伽玛变换
def gamma(img, c, v):
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        lut[i] = c * i ** v
    output_img = cv2.LUT(img, lut) #像素灰度值的映射
    output_img = np.uint8(output_img+0.5)
    return output_img

def gary_process(img):
    '''
    :param img:
    :return:
    '''
    # 图像灰度转换
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #获取图像高度和宽度
    height = img.shape[0]
    width = img.shape[1]

    #创建一幅图像
    maxgrayimg = np.zeros((height, width, 3), np.uint8)
    avegrayimg = np.zeros((height, width, 3), np.uint8)
    weightgrayimg = np.zeros((height, width, 3), np.uint8)
    grayupimg = np.zeros((height, width), np.uint8)
    contrastenhancementimg = np.zeros((height, width), np.uint8)
    graynonlinear = np.zeros((height, width), np.uint8)

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


    # 图像灰度上移变换 DB=DA+50
    for i in range(height):
        for j in range(width):

            if (int(grayImage[i, j] + 50) > 255):
                gray = 255
            else:
                gray = int(grayImage[i, j] + 50)
            grayupimg[i, j] = np.uint8(gray)
    # 图像对比度增强变换 DB=DA*1.5
    for i in range(height):
        for j in range(width):

            if (int(grayImage[i, j] * 1.1) > 255):
                gray = 255
            else:
                gray = int(grayImage[i, j] * 1.1)
            contrastenhancementimg[i, j] = np.uint8(gray)

    # 图像灰度非线性变换：DB=DA×DA/255
    for i in range(height):
        for j in range(width):
            gray = int(grayImage[i, j]) * int(grayImage[i, j]) / 255
            graynonlinear[i, j] = np.uint8(gray)

    # 图像灰度对数变换
    logimg = log(42, img)
    # 图像灰度伽玛变换
    gammaimg = gamma(img, 0.00000005, 4.0)
    # 调用matplotlib显示处理结果
    titles = ['image', 'maxgrayimg', 'avegrayimg', 'weightgrayimg','grayupresult',
              'contrastenhancementimg','graynonlinear','logimg','gammaimg']
    images = [img, maxgrayimg, avegrayimg, weightgrayimg,grayupimg,
              contrastenhancementimg,graynonlinear,logimg,gammaimg]
    for i in range(9):
        plt.subplot(3, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()