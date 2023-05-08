import torch
import torchvision
from torch import nn
from torch.nn import functional as F, Sequential, ReLU, MaxPool2d, Conv2d, AdaptiveAvgPool2d, Dropout
from GaborNet import GaborConv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from PIL import Image
from torchvision import transforms
from GaborModels import *

train_data = torchvision.datasets.CIFAR10(root="data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# #搭建网络：在 VGG16 的基础上将前两层变化为Gabor卷积网络-》论文中
# class GCN_test(nn.Module):
#     def __init__(self):
#         super(GCN_test, self).__init__()
#         self.features=nn.Sequential(
#             GaborConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),padding=(1,1)),
#             nn.ReLU(inplace=True),
#             GaborConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn. Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#         )
#         self.avgpool=nn.AdaptiveAvgPool2d(output_size=(7, 7))
#         self.classifier=nn.Sequential(
#             nn.Flatten(),# 展开，将图片转化为一维线性
#             nn.Linear(in_features=25088, out_features=4096, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5, inplace=False),
#             nn.Linear(in_features=4096, out_features=4096, bias=True),
#             nn. ReLU(inplace=True),
#             nn.Dropout(p=0.5, inplace=False),
#             nn.Linear(in_features=4096, out_features=10, bias=True), #在 CIFAR10上训练
#         )
#
#     def forward(self, x):
#         x=self.features(x)
#         x=self.avgpool(x)
#         x=self.classifier(x)
#         return x
# #创建网络 1
# gcn_test=GCN_test()
#创建网络 2
#gcn_test_2=Gabor_vgg()
#创建网络 3
gcn_test_3=GCN13()
#创建网络 4
device=torch.device('cuda'if torch.cuda.is_available() else "cpu")#电脑主机的选择
VGG16=vgg16(True, progress=True,num_classes=10)#定于分类的类别

#测试数据
# test_pth="Dataset/train/ants/0013035.jpg"
# img_test=Image.open(test_pth)
# tensor_trans=torchvision.transforms.ToTensor()   #利用创建好的工具将图片文件转化为tensor类型
# img_tensor=tensor_trans(img_test)
# print(img_tensor.shape)
# img=torch.reshape((img_tensor),(-1,3,32,32))
# print(img.shape)
# gcn_test(img)
# summary(gcn_test,(3,32,32))  #输入图片经过每层后尺寸的变化

#
# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
#optimizer = torch.optim.SGD(gcn_test.parameters(),learning_rate)  #网络1
#optimizer = torch.optim.SGD(gcn_test_2.parameters(),learning_rate)  #网络2
# optimizer = torch.optim.SGD(gcn_test_3.parameters(),learning_rate)  #网络3
optimizer = torch.optim.SGD(VGG16.parameters(),learning_rate)  #网络3


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("gcn_train")
#
for i in range(epoch):  #大循环：训练轮数
    print("--------第 {} 轮训练开始-------".format(i+1))
    # 训练步骤开始
    #gcn_test.train()
    #gcn_test_2.train()
    #gcn_test_3.train()
    VGG16.train()
    for data in train_dataloader: #小循环：遍历训练集中的每一张图片
        imgs, targets = data
        optimizer.zero_grad()
        # outputs =gcn_test(imgs)
        # outputs = gcn_test_2(imgs)
        # outputs = gcn_test_3(imgs)
        outputs = VGG16(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  #每百次输出一次结果
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)  #绘制损失函数图（训练集上）

       # 测试步骤开始
       #  gcn_test.eval()
        #gcn_test_2.eval()
        # gcn_test_3.eval()
        VGG16.eval()
        total_test_loss = 0
        total_accuracy = 0  #测试网络模型的精确度

    with torch.no_grad():  # 测试每一轮训练好的网络模型，故不对当前模型进行优化，不使用梯度
        for data in test_dataloader:
            imgs, targets = data
            # outputs =gcn_test(imgs)
            #outputs = gcn_test_2(imgs)
            # outputs = gcn_test_3(imgs)
            outputs = VGG16(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = ((outputs.argmax(1) == targets).sum()).clone().detach().cpu().numpy() # argmax(1)表示按行进行读取
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    #torch.save(gcn_test,"Result_GCN_test/GCN_test{}.pth".format(i))  #保存当前训练好的网络
    #print("模型已保存")
    #torch.load("Result_GCN_test/GCN_test{}.pth".format(i))  # 加载已保存的模型
    #torch.save(gcn_test_2,"Result_GCN_test2/GCN_test{}.pth".format(i))  #保存当前训练好的网络
    #print("模型已保存")
    #torch.save(gcn_test_3, "Result_GCN_test3/GCN_test{}.pth".format(i))
    torch.save(vgg16, "Result_VGG16_test/VGG16_test{}.pth".format(i))
    print("模型已保存")

writer.close()


    

