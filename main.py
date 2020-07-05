from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from lenet import LeNet


def normalize_image(images):
    ''' 对图像做归一化处理 '''
    result=np.zeros(images.shape)
    for i in range(0,images.shape[0]):
        result[i]=images[i]/(np.max(images[i])-np.min(images[i]))
        #result[i]-=0.5
    images=result
    return images

def one_hot_labels(labels):
    '''
    将labels 转换成 one-hot向量
    eg:  label: 3 --> [0,0,0,1,0,0,0,0,0,0]
    '''
    one_hot=np.array([0,0,0,0,0,0,0,0,0,0])
    ONE_HOT=np.tile(one_hot,(labels.shape[0],1))#tile方法可以将一个向量扩展成n个重复的该向量组成的矩阵
    for i in range(0,labels.shape[0]):#i的范围是0到labels.shape[0]-1
        ONE_HOT[i,labels[i]]=1
    labels=ONE_HOT#将labels转为一个labels.shape[0]*10的矩阵，对于每一个原来的labels[i],现在是labels[i,:],显示one_hot横向量
    return labels


def main():
    # image shape: N x H x W, pixel [0, 255]
    # label shape: N x 10
    ###
    np.seterr(invalid='ignore')#这一行是由于numpy会对relu和softmax的运算显示可能无效警告，然而这个风险是可忽略的
    ###
    with np.load('mnist.npz', allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']

    plt.imshow(x_train[59999], cmap='gray')
    plt.show()
    print(x_train.shape, x_train[0].max(), x_train[0].min()) #(60000, 28, 28) 255 0 5
    print(x_test.shape, x_test[0].max(), x_test[0].min()) #(10000, 28, 28) 255 0 7

    x_train = normalize_image(x_train)
    x_test = normalize_image(x_test)
    y_train = one_hot_labels(y_train)
    y_test = one_hot_labels(y_test)

    net = LeNet()

    net.fit(x_train[:1000], y_train[:1000], x_test[:100], y_test[:100], epoches=10, batch_size=16, lr=1e-6)

    accu = net.evaluate(x_test[:100], labels=y_test[:100])
    print("final accuracy {}".format(accu))


if __name__ == "__main__":
    main()


