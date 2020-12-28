import torch
import torch.nn as nn
from torch.nn import functional as F
#Lenet5网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #1 input image,6 output channels,5*5 square convolution

        #网络层
        #(w+2*p-k)/strides+1
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        #w+2*
        #an affine operation:y=wx+b
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        print('ss')
    def forward(self,x):
        # Max pooling over a(2,2) window
        #max_pooling层
        #((32-5+1)/1+1)/2=14(向下取整）
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # If the size is a square you can only specify a single number
        # ((14-5+1)/1+1)/2=5
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self,x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#一个模型可训练的参数可以通过调用 net.parameters() 返回：
params = list(net.parameters())
print(len(params))
print('conv1:',params[3].size()) #conv1.weight


input = torch.randn(1,1,32,32)
out = net(input)
print(out)

#把所有参数梯度缓存器置零，用随机的梯度来反向传播
net.zero_grad()
out.backward(torch.randn(1,10))


'''
在此，我们完成了：

1.定义一个神经网络

2.处理输入以及调用反向传播
还剩下：

1.计算损失值

2.更新网络中的权重

损失函数

一个损失函数需要一对输入：模型输出和目标，然后计算一个值来评估输出距离目标有多远。

有一些不同的损失函数在 nn 包中。一个简单的损失函数就是 nn.MSELoss ，这计算了均方误差。
'''

output = net(input)
target = torch.randn(10)#a dummpy target,for example
target = target.view(1,-1)#make it the same sample as out put
criterion = nn.MSELoss()

loss = criterion(output,target)
print(loss)

'''
为了实现反向传播损失，我们所有需要做的事情仅仅是使用 loss.backward()。你需要清空现存的梯度，要不然帝都将会和现存的梯度累计到一起。

现在我们调用 loss.backward() ，然后看一下 con1 的偏置项在反向传播之前和之后的变化。
'''

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


'''
唯一剩下的事情就是更新神经网络的参数。

更新神经网络参数：

最简单的更新规则就是随机梯度下降。
weight = weight - learning_rate * gradient

如何实现呢：
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
    
    
尽管如此，如果你是用神经网络，你想使用不同的更新规则，类似于 SGD, Nesterov-SGD, Adam, RMSProp, 等。为了让这可行，我们建立了一个小包：torch.optim 实现了所有的方法。使用它非常的简单。

'''
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update