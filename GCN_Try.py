import torch
import torchvision
from torch import nn
from torch.nn import functional as F, Sequential, ReLU, MaxPool2d, Conv2d, AdaptiveAvgPool2d, Dropout
from GaborNet import GaborConv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from GaborModels import *

# classes = (1:'plane',2:'car',3:'bird',4:'cat',5:'deer',6:'dog',7:'frog',8:'horse',9:'ship',10:'truck')
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

#数据集所分类数目
Class=10

# 训练的轮数
epoch = 10

# #创建网络 1
# gcn_test=GCN_test()
#创建网络 2
#gcn_test_2=Gabor_vgg()
#创建网络 3
# gcn_test_3=GCN13()
#创建网络 4
VGG16=vgg16(True, progress=True,num_classes=10)#定于分类的类别
#创建网络 5
gcn_test_4=gcn13_2(Class)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

#学习率
warm_up = 0.05
decay_rate = 0.95
decay_steps = 20
# 自定义学习率衰减算法
def self_adjust_learning_rate(optimizer, train_sum):
    warm_up_step = int(epoch * warm_up)
    if train_sum < warm_up_step:
        lr = 0.1 * LR + (LR - 0.1 * LR) / warm_up_step * train_sum
    else:
        lr = LR * (decay_rate ** ((train_sum - warm_up_step) // decay_steps))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
LR = 1e-2  #初始学习率数值
# 优化器
#optimizer = torch.optim.SGD(gcn_test.parameters(),learning_rate)  #网络1
#optimizer = torch.optim.SGD(gcn_test_2.parameters(),learning_rate)  #网络2
# optimizer = torch.optim.SGD(gcn_test_3.parameters(),learning_rate)  #网络3
# optimizer = torch.optim.SGD(VGG16.parameters(),lr=LR)  #网络4
optimizer = torch.optim.SGD(gcn_test_4.parameters(),lr=LR)  #网络5

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0


# 添加tensorboard
writer = SummaryWriter("gcn_train")
#
for i in range(epoch):  #大循环：训练轮数
    print("--------第 {} 轮训练开始-------".format(i+1))
    # 训练步骤开始
    #gcn_test.train()
    #gcn_test_2.train()
    #gcn_test_3.train()
    # VGG16.train()
    gcn_test_4.train()
    for data in train_dataloader: #小循环：遍历训练集中的每一张图片
        imgs, targets = data
        optimizer.zero_grad()
        # outputs =gcn_test(imgs)
        # outputs = gcn_test_2(imgs)
        # outputs = gcn_test_3(imgs)
        # outputs = VGG16(imgs)
        outputs = gcn_test_4(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        lr = self_adjust_learning_rate(optimizer, i)  # 自定义学习率衰减算法

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:  #每百次输出一次结果
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)  #绘制损失函数图（训练集上）

       # 测试步骤开始
       #  gcn_test.eval()
        #gcn_test_2.eval()
        # gcn_test_3.eval()
        # VGG16.eval()
        gcn_test_4.eval()
        total_test_loss = 0
        total_accuracy = 0  #测试网络模型的精确度

    with torch.no_grad():  # 测试每一轮训练好的网络模型，故不对当前模型进行优化，不使用梯度
        for data in test_dataloader:
            imgs, targets = data
            # outputs =gcn_test(imgs)
            #outputs = gcn_test_2(imgs)
            # outputs = gcn_test_3(imgs)
            # outputs = VGG16(imgs)
            outputs=gcn_test_4(imgs)
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
    # torch.save(vgg16, "Result_VGG16_test/VGG16_test{}.pth".format(i))
    torch.save(gcn_test_4, "Result_GCN_test4/GCN_test{}.pth".format(i))
    print("模型已保存")
    # torch.load("Result_VGG16_test/VGG16_test{}.pth".format(i))  # 加载已保存的模型

writer.close()


    

