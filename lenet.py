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

    def forward(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, z):
        return self.forward(z) * (1 - self.forward(z))

#Relu类
class Relu():
    def __init__(self):
        pass

    def forward(self,x):
        return (x>=0)*x
        
    def backward(self,z):
        return (z>0)*1#z为函数值
#Conv类
class Conv():
    def __init__(self,num,size,channel,activation):#这里默认stride为1
        self.num=num
        self.size=size
        self.channel=channel
        self.activation=activation
        self.kernel=np.random.uniform(-1,1,(num,channel,size,size))#N*C*W*H
    def conv_operation(self,post,kernel):#post C*H*W kernel(N,C,H,W)
        C,H,W=post.shape
        S_K,C_K=(kernel.shape[2],kernel.shape[0])
        kernel=kernel.reshape((C_K,C*(S_K**2)))
        mat=np.zeros((C*(S_K**2),(H-(S_K-1))*(W-(S_K-1))))
        for i in range(H-(S_K-1)):
                for j in range(W-(S_K-1)):
                    mat[:,i*j]+=post[:,i:(i+S_K),j:(j+S_K)].reshape(C*S_K**2)
        return (kernel.dot(mat)).reshape(C_K,H-(S_K-1),W-(S_K-1))
    def forward(self,origin):
        if(len(origin.shape)==2):#防止通道数为1使shape不规范
            origin=origin.reshape((1,origin.shape[0],origin.shape[1]))
        self.post=origin#C*H*W
        
        self._forward=self.activation.forward(self.conv_operation(self.post,self.kernel))
        return self._forward#N*H*W
    def backward(self,error,learning_rate):
        self.db=error*self.activation.backward(self._forward)#N*H*W
        self.dw=np.zeros(self.kernel.shape)
        for p in range(self.post.shape[0]):#self.post与self.db的卷积
            self.dw[:,p,:,:]+=self.conv_operation(self.post[p,:,:].reshape(1,self.post.shape[1],self.post.shape[2]),self.db.reshape((self.db.shape[0],1,self.db.shape[1],self.db.shape[2])))
        self.dB=np.pad(self.db,((0,0),(self.size-1,self.size-1),(self.size-1,self.size-1)))#zero_pad=size-1
        self.pi_kernel=np.rot90(self.kernel,2)#卷积核的旋转
        self.backerror=self.conv_operation(self.dB,self.pi_kernel.transpose(1,0,2,3))#self.dB与self.pi_kernel的卷积
        self.kernel-=learning_rate*self.dw#更新权值
        return self.backerror
#Avgpool类
class Avgpool():
    def __init__(self,size):
        self.size=size
    def forward(self,origin):
        self.input=origin#origin的维度是C*H*W
        C=origin.shape[0]
        H=int(origin.shape[1]/self.size)
        W=int(origin.shape[2]/self.size)#OUTPUT SIZE C*H*W
        self.col=np.zeros((C,H*W,self.size**2))
        self.mean_kernel=np.ones((self.size**2,1))/(self.size**2)
        for i in range(H):
            for j in range(W):
                self.col[:,i*W+j,:]=origin[:,self.size*i:(self.size*i+self.size),self.size*j:(self.size*j+self.size)].reshape(C,self.size**2)
        self._forward=self.col.dot(self.mean_kernel).reshape(C,H,W)
        return self._forward
    def backward(self,error):
        self.backerror=error.repeat(self.size,axis=1).repeat(self.size,axis=2)/self.size**2
        return self.backerror
#Softmax类
class Softmax():
    def __init__(self,length):
        self.length=length
    def forward(self,vector):#vector是一个长度为length的行向量
        self.post=vector
        self._forward=np.array([1/(np.sum(np.exp(vector-val))) for val in vector])#防止溢出
        return self._forward
    def backward(self,error):#error是行向量
        self.backerror=((self._forward.T)*(np.eye(self.post.shape[0])-self._forward)).dot(error.T)
        return self.backerror
#Fc类
class Fc():
    def __init__(self,innum,outnum,activation):
        self.innum=innum
        self.outnum=outnum
        self.activation=activation
        self.w=np.random.uniform(-1,1,(innum,outnum))#w矩阵的第i行第j列是上一层的第i个元素到这一层的第j个元素的权重
    def forward(self,origin):
        self.post=origin#post应是一个行向量
        self._forward=self.activation.forward(self.post.dot(self.w))
        return self._forward
    def backward(self,error,learning_rate):
        self.miderror=error*self.activation.backward(self._forward)
        self.dw=self.post.reshape((self.post.shape[0],1)).dot(self.miderror.reshape(1,self.miderror.shape[0]))
        self.backerror=(self.w.dot(self.miderror.T)).T
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
        self.activation=Relu()
        self.conv_1=Conv(6,5,1,self.activation)
        self.conv_2=Conv(16,5,6,self.activation)
        self.avg_1=Avgpool(2)
        self.avg_2=Avgpool(2)
        self.fc_1=Fc(256,128,self.activation)
        self.fc_2=Fc(128,64,self.activation)
        self.fc_3=Fc(64,10,self.activation)
        self.softmax=Softmax(10)
        
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
        self.conv_1_forward=[]
        self.avg_1_forward=[]
        self.conv_2_forward=[]
        self.avg_2_forward=[]
        self.fc_1_forward=[]
        self.fc_2_forward=[]
        self.fc_3_forward=[]
        self.softmax_forward=[]

        for i in range(x.shape[0]):
            
            self.conv_1_forward.append(self.conv_1.forward(x[i]))
            self.avg_1_forward.append(self.avg_1.forward(self.conv_1_forward[i]))
            self.conv_2_forward.append(self.conv_2.forward(self.avg_1_forward[i]))
            self.avg_2_forward.append(self.avg_2.forward(self.conv_2_forward[i]))
            self.fc_1_forward.append(self.fc_1.forward(self.avg_2_forward[i].reshape(256)))#flatten
            self.fc_2_forward.append(self.fc_2.forward(self.fc_1_forward[i]))
            self.fc_3_forward.append(self.fc_3.forward(self.fc_2_forward[i]))
            self.softmax_forward.append(self.softmax.forward(self.fc_3_forward[i]))
        return self.softmax_forward#是一个列表包含所有样本的one-hot

    def backward(self, error, lr=1.0e-3):
        """根据error，计算梯度，并更新model中的权值
        Arguments:
            error {np array} -- 即计算得到的loss结果
            lr {float} -- 学习率，可以在代码中设置衰减方式
        """
        self.conv_1_backward=[]
        self.avg_1_backward=[]
        self.conv_2_backward=[]
        self.avg_2_backward=[]
        self.fc_1_backward=[]
        self.fc_2_backward=[]
        self.fc_3_backward=[]
        self.softmax_backward=[]
        
        for i in range(error.shape[0]):#这里error的shape为（N,10)
            self.softmax_backward.append(self.softmax.backward(error[i]))
            self.fc_3_backward.append(self.fc_3.backward(self.softmax_backward[i],lr))
            self.fc_2_backward.append(self.fc_2.backward(self.fc_3_backward[i],lr))
            self.fc_1_backward.append(self.fc_1.backward(self.fc_2_backward[i],lr))

            self.avg_2_backward.append(self.avg_2.backward(self.fc_1_backward[i].reshape((16,4,4))))
            self.conv_2_backward.append(self.conv_2.backward(self.avg_2_backward[i],lr))
            
            self.avg_1_backward.append(self.avg_1.backward(self.conv_2_backward[i]))
            self.conv_1_backward.append(self.conv_1.backward(self.avg_1_backward[i],lr))
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
        counter=0
        result=self.forward(x)
        for i in range(x.shape[0]):    
            if(np.sqrt(np.sum((result[i]-labels[i])**2))<=1e-2):
                counter+=1
        return counter/x.shape[0]

    def data_augmentation(self, images):
        '''
        数据增强，可选操作，非强制，但是需要合理
        一些常用的数据增强选项： ramdom scale， translate， color(grayscale) jittering， rotation, gaussian noise,
        这一块儿允许使用 opencv 库或者 PIL image库
        比如把6旋转90度变成了9，但是仍然标签为6 就不合理了
        '''
        return images
    def compute_loss(self,pred,label):#loss用均方和
        return 2*(pred-label)
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
            # 1. 一次forward，backward肯定不能是所有的图像一起,
            因此需要根据 batch_size 将 train_image, 和 train_label 分成: [ batch0 | batch1 | ... | batch_last]
            '''
            batch_images = [] # 请实现 step #1
            batch_labels = [] # 请实现 step #1
            max_size_int=int(train_image.shape[0]/batch_size)
            for i in range(max_size_int):
                batch_images.append(train_image[batch_size*i:batch_size*(i+1)])
                batch_labels.append(train_label[batch_size*i:batch_size*(i+1)])
            if(max_size_int*batch_size<train_image.shape[0]):
                batch_images.append(train_image[batch_size*max_size_int:])
                batch_labels.append(train_label[batch_size*max_size_int:])
            last = time.time() #计时开始
            for imgs, labels in zip(batch_images, batch_labels):
                '''
                这里我只是给了一个范例， 大家在实现上可以不一定要按照这个严格的 2,3,4步骤
                我在验证大家的模型时， 只会在main中调用 fit函数 和 evaluate 函数。
                2. 做一次forward，得到pred结果  eg. pred = self.forward(imgs)
                3. pred 和 labels做一次 loss eg. error = self.compute_loss(pred, labels)
                4. 做一次backward， 更新网络权值  eg. self.backward(error, lr=1e-3)
                '''
                pre=time.time()
                pred=self.forward(imgs)
                error=self.compute_loss(pred, labels)
                self.backward(error,lr)
                print(pred[0])
                print(time.time()-pre)
            duration = time.time() - last
            sum_time += duration

            if epoch % 5 == 0:
                accuracy = self.evaluate(test_image, test_label)
                print("epoch{} accuracy{}".format(epoch, accuracy))
                accuracies.append(accuracy)
            print(epoch)
        avg_time = sum_time / epoches
        return avg_time, accuracies


