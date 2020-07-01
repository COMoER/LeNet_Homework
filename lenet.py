from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time



# Example Sigmoid
# 这个类中包含了 forward 和backward函数
class Sigmoid():
    def __init__(self):
        pass

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))

#Relu类
class Relu():
    def __init__(self):
        pass

    def forward(self, x):
        return (x>=0)*x
        
    def backward(self, z):
        return (z>0)*1#z为函数值
#Conv类
class Conv():#conv的语法还需要改进，忽略了层数channel
    def __init__(self,num,size,activation):
        self.num=num
        self.size=size
        self.activation=activation
        self.conv=np.random.random((num,size,size))
    def forward(self,origin):
        self.forward=np.zeros(self.num,origin.shape[0]-self.size+1,origin.shape[1]-self.size+1)
        for i in range(0,self.num):
            for j in range(0,origin.shape[0]-(self.size-1)):
                for k in range(0,origin.shape[1]-(self.size-1)):
                    sum=np.sum(origin[j:(j+self.size-1),k:(k+self.size-1)]*self.conv[i])
                    self.forward[i,j,k]=self.activation.forward(sum)
        return self.forward
    
#Avgpool类
class Avgpool():#同样忽略了层数
    def __init__(self,size):
        self.size=size
    def forward(self,origin):
        self.forward=np.zeros((origin.shape[0]/2,origin.shape[1]/2))
        for i in range(0,origin.shape[0]/2):
            for j in range(0,origin.shape[1]/2):
                self.forward[i,j]=np.mean(origin[2*i:(2*i+1),2*j:(2*j+1)])
        return self.forward

#Fc类
class Fc():
    def __init__(selk,innum,outnum,activation):
        self.innum=innum
        self.outnum=outnum
        self.activation=activation
        self.w=np.random.random((innum,outnum))
    def forward(self,origin):
        self.post=origin
        self.forward=self.activation.forward(self.post.dot(w))
        return self.forward
    def backward(self,error,learning_rate):
        self.miderror=error*self.activation.backward(self.forward)
        self.dw=self.post.T.dot(self.miderror)
        self.backerror=self.w.dot(self.miderror.T)
        self.w-=learning_rate*self.dw
        return self.backerror
## 在原 LeNet-5上进行少许修改后的网路结构
"""
conv1: in_channels: 1, out_channel:6, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool1: in_channels: 6, out_channels:6, kernel_size = (2x2), stride=2
conv2: in_channels: 6, out_channel:16, kernel_size=(5x5), pad=0, stride=1, activation: relu
avgpool2: in_channels: 16, out_channels:16, kernel_size = (2x2), stride=2
flatten
fc1: in_channel: 256, out_channels: 128, activation: relu
fc2: in_channel: 128, out_channels: 64, activation: relu
fc3: in_channel: 64, out_channels: 10, activation: relu
softmax:

tensor: (1x28x28)   --conv1    -->  (6x24x24)
tensor: (6x24x24)   --avgpool1 -->  (6x12x12)
tensor: (6x12x12)   --conv2    -->  (16x8x8)
tensor: (16x8x8)    --avgpool2 -->  (16x4x4)
tensor: (16x4x4)    --flatten  -->  (256)
tensor: (256)       --fc1      -->  (128)
tensor: (128)       --fc2      -->  (64)
tensor: (64)        --fc3      -->  (10)
tensor: (10)        --softmax  -->  (10)
"""


class LeNet(object):
    def __init__(self):
        '''
        初始化网路，在这里你需要，声明各Conv类， AvgPool类，Relu类， FC类对象，SoftMax类对象
        并给 Conv 类 与 FC 类对象赋予随机初始值
        注意： 不要求做 BatchNormlize 和 DropOut, 但是有兴趣的可以尝试
        '''
        self.activation=Relu
        self.conv[0]=Conv(6,5,self.activation)
        self.conv[1]=Conv(16,5,self.activation)
        self.avg[0]=Avgpool(2)
        self.avg[1]=Avgpool(2)
        
        
        print("initialize")

    def init_weight(self):
        pass

    def forward(self, x):
        """前向传播
        x是训练样本， shape是 B,C,H,W
        这里的C是单通道 c=1 因为 MNIST中都是灰度图像
        返回的是最后一层 softmax后的结果
        也就是 以 One-Hot 表示的类别概率

        Arguments:
            x {np.array} --shape为 B，C，H，W
        """
        return 0

    def backward(self, error, lr=1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        pass

    def evaluate(self, x, labels):
        """
        x是测试样本， shape 是BCHW
        labels是测试集中的标注， 为one-hot的向量
        返回的是分类正确的百分比

        在这个函数中，建议直接调用一次forward得到pred_labels,
        再与 labels 做判断

        Arguments:
            x {np array} -- BCWH
            labels {np array} -- B x 10
        """
        return 0

    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        return images

    def fit(
        self,
        train_image,
        train_label,
        test_image = None,
        test_label = None,
        epoches = 10,
        batch_size = 16,
        lr = 1.0e-3
    ):
        sum_time = 0
        accuracies = []

        for epoch in range(epoches):

            ## 可选操作，数据增强
            train_image = self.data_augmentation(train_image)
            ## 随机打乱 train_image 的顺序， 但是注意train_image 和 test_label 仍需对应
            '''
            # 1. 一次forward，bachword肯定不能是所有的图像一起,
            因此需要根据 batch_size 将 train_image, 和 train_label 分成: [ batch0 | batch1 | ... | batch_last]
            '''
            batch_images = [] # 请实现 step #1
            batch_labels = [] # 请实现 step #1

            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                pass
            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)

        avg_time = sum_time / epoches
        return avg_time, accuracies


