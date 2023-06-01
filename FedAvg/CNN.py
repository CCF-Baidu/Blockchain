import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# 超参数
batch_size = 64
learning_rate = 0.02
num_eporches =20
# 数据准备
data_ft = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])

train_dataset = datasets.MNIST("./data",train=True,transform=data_ft,download=True)
test_dataset = datasets.MNIST("./data",train=False,transform=data_ft)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)


# CNN模型
# 建立四个卷积层网络 、两个池化层 、 1个全连接层
# 第一层网络中，为卷积层，将28*28*1的图片，转换成16*26*26
# 第二层网络中，包含卷积层和池化层。将16*26*26 -> 32*24*24,并且池化成cheng32*12*12
# 第三层网络中，为卷积层，将32*12*12 -> 64*10*10
# 第四层网络中，为卷积层和池化层，将 64*10*10 -> 128*8*8,并且池化成128*4*4
# 第五次网络为全连接网络

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # 第二次卷积的输出拉伸为一行
        x = self.fc(x)
        return x


model = CNN()
if torch.cuda.is_available():
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=learning_rate)
# 训练模型
epoch = 0
lossp=[]
for data in train_loader:
    img, label = data

    # 与全连接网络不同，卷积网络不需要将所像素矩阵转换成一维矩阵
    # img = img.view(img.size(0), -1)

    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    else:
        img = Variable(img)

        label = Variable(label)

    out = model(img)
    loss = criterion(out, label)
    print_loss = loss.data.item()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    if epoch % 50 == 0:
        print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
    lossp.append(loss.data.item())
# plot loss curve
plt.figure()
plt.plot(range(len(lossp)), lossp)
plt.ylabel('train_loss')
plt.show()
plt.savefig('cnn.png')