import torch
import torchvision
from torch import nn
from GaborNet import GaborConv2d
from torch.nn import functional as F, Sequential, ReLU, MaxPool2d, Conv2d, AdaptiveAvgPool2d, Dropout
from torchsummary import summary
from torch.hub import load_state_dict_from_url


#搭建网络1---在VG16基础上改动
class GCN_test(nn.Module):
    def __init__(self):
        super(GCN_test, self).__init__()
        self.features=nn.Sequential(
            GaborConv2d(in_channels=3, out_channels=64, kernel_size=(3, 3),padding=(1,1)),
            nn.ReLU(inplace=True),
            GaborConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn. Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
        self.avgpool=nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier=nn.Sequential(
            nn.Flatten(),# 展开，将图片转化为一维线性
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn. ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=10, bias=True), #在 CIFAR10上训练
        )

    def forward(self, x):
        x=self.features(x)
        x=self.avgpool(x)
        x=self.classifier(x)
        return x


#搭建网络 2，修改 VGG16
Gabor_vgg=torchvision.models.vgg16(pretrained=True)  #预训练网络
# print(Gabor_vgg)
Gabor_vgg.features[0]=GaborConv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Gabor_vgg.features[2]=GaborConv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Gabor_vgg.classifier[6]=nn.Linear(in_features=4096, out_features=10, bias=True)
# print(Gabor_vgg)
# #查看每层数据变化
# summary(Gabor_vgg,(3,32,32))


#搭建网络 3，GaborConv13
class GCN13(nn.Module):
    def __init__(self):
        super(GCN13, self).__init__()
        self.gcon1=Sequential(
            GaborConv2d(3,64,(5,5),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64), #对批次内特征进行归一化处理
            nn.ReLU(),
        )
        self.con2=Sequential(
            nn.Conv2d(64,64,(3,3),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.con3=Sequential(
            nn.Conv2d(64,128,(3,3),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.con4=Sequential(
            nn.Conv2d(128,128,(3,3),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.con5=Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier=Sequential(
            nn.Flatten(),
            nn.Linear(25088,4096,True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096,True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10, True),
        )
    def forward(self,x):
        x=self.gcon1(x)
        x=self.con2(x)
        x=self.con3(x)
        x=self.con4(x)
        x=self.con5(x)
        x=self.classifier(x)
        return x
# #查看每层数据变化
# gcn13_test=GCN13()
# summary(gcn13_test,(3,32,32))


#搭建网络 4，改动原GaborConv13，第一层为 9*9，第一个池化层为 4*4
class GCN13_2(nn.Module):
    def __init__(self):
        super(GCN13_2, self).__init__()
        self.gcon1=Sequential(
            GaborConv2d(3,64,(9,9),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64), #对批次内特征进行归一化处理
            nn.ReLU(),
        )
        self.con2=Sequential(
            nn.Conv2d(64,64,(3,3),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.con3=Sequential(
            nn.Conv2d(64,128,(3,3),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.con4=Sequential(
            nn.Conv2d(128,128,(3,3),stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.con5=Sequential(
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier=Sequential(
            nn.Flatten(),
            nn.Linear(12800,4096,True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096,True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10, True),
        )
    def forward(self,x):
        x=self.gcon1(x)
        x=self.con2(x)
        x=self.con3(x)
        x=self.con4(x)
        x=self.con5(x)
        x=self.classifier(x)
        return x

# #查看每层数据变化
# gcn13_2_test=GCN13_2()
# summary(gcn13_2_test,(3,32,32))

#搭建网络 4， VGG16
model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",

}#权重下载网址


class VGG16(nn.Module):
    def __init__(self, features, num_classes = 1000, init_weights= True, dropout = 0.5):
        super(VGG16,self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))#AdaptiveAvgPool2d使处于不同大小的图片也能进行分类
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),#完成4096的全连接
            nn.Linear(4096, num_classes),#对num_classes的分类
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)#对输入层进行平铺，转化为一维数据
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm = False):#make_layers对输入的cfg进行循环
    layers = []
    in_channels = 3
    for v in cfg:#对cfg进行输入循环,取第一个v
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]#把输入图像进行缩小
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)#输入通道是3，输出通道64
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],

}

def vgg16(pretrained=False, progress=True,num_classes=10):
    model = VGG16(make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'],model_dir='./model' ,progress=progress)#预训练模型地址
        model.load_state_dict(state_dict)
    if num_classes !=1000:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),#随机删除一部分不合格
            nn.Linear(4096, 4096),
            nn.ReLU(True),#防止过拟合
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    return model
if __name__=='__main__':
    in_data=torch.ones(1,3,224,224)
    net=vgg16(pretrained=False, progress=True,num_classes=10)
    out=net(in_data)
    print(out)
