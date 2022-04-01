# --------------------------------------------------------------------------------------------------------------------------------------------------
# 修改者：       赵敏琨
# 日期：       2021年6月
# 说明：       计算自己数据集的均值和标准差
# --------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import os

if __name__ == '__main__':
    img_h, img_w = 112, 92  # 根据自己数据集图片大小调整
    means, stdevs = [], []
    img_list = []

    imgs_path = '../matlabProject/eigfaces/train_set/'
    imgs_path_list = os.listdir(imgs_path)
    print(imgs_path_list)

    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(imgs_path, item))
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)

    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))