##导入库
import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda : 0" if torch.cuda.is_available() else "cpu")
import numpy as np
import time
start_time=time.time()
##建立网络
class Net(nn.Module):
    def __init__(self):
         super(Net,self).__init__()
         self.hidden_layer1 = nn.Linear(2,5)
         self.hidden_layer2 = nn.Linear(5, 5)
         self.hidden_layer3 = nn.Linear(5, 5)
         self.hidden_layer4 = nn.Linear(5, 5)
         self.hidden_layer5 = nn.Linear(5, 5)
         self.output_layer=nn.Linear(5,1)
    def forward(self,x,t):
         inputs=torch.cat([x,t],axis=1)##x和t原本分别是 1 列的张量，那么拼接后就变成了一个具有两列的张量
         layer1_out=torch.sigmoid(self.hidden_layer1(inputs))
         layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
         layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
         layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
         layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
         output=self.output_layer(layer5_out)##输出层不使用激活函数
         return output
##创立模型
net=Net()
net=net.to(device)##将创建的神经网络模型移动到指定的设备上
mse_cost_function=torch.nn.MSELoss()#定义了一个均方误差（Mean Squared Error，MSE）损失函数。
optimizer = torch.optim.Adam(net.parameters())##创建优化器，使用Adam优化算法，parameters返回模型中可训练的参数
def f(x,t,net):
    u=net(x,t)##使用传入的神经网络模型net对输入的x和t进行前向传播，得到网络的输出u。
   ##计算u关于x的偏导数。首先对u的所有元素求和（u.sum()），然后使用torch.autograd.grad函数计算关于x的梯度。create_graph=True表示创建计算图，以便在后续的计算中可以继续进行自动求导。
    # 返回的结果是一个包含梯度的元组，取第一个元素[0]得到关于x的偏导数u_x。
    u_x=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
    u_t=torch.autograd.grad(u.sum(),t,create_graph=True)[0]
    pde=u_x-2*u_t-u#根据给定的偏微分方程f = du/dx - 2du/dt - u，计算偏微分方程的值。这里使用前面计算得到的u_x和u_t，
    # 以及网络输出u来计算偏微分方程的值。
    return pde
##生成x，t的数据点，已知偏微分方程的边界条件为 u(x,0)=6e^(-3x)
x_bc=np.random.uniform(low=0.0,high=2.0,size=(500,1))##在0——2之间产生500个样本
t_bc=np.zeros((500,1))
u_bc=6*np.exp(-3*x_bc)

##训练阶段
iteration=20000
ite=[]
lo=[]
previous_validation_loss=99999999.0##初始化一个较大的数值作为上一次验证集的损失，用于后续可能的比较
for epoch in range(iteration):
    optimizer.zero_grad()##每次迭代开始时，清零优化器中的梯度
    ##将 numpy 数组 x_bc 转换为 PyTorch 的张量，并封装为 Variable，
    # 设置 requires_grad=False 表示这个张量在计算中不需要计算梯度，然后将其移动到指定的设备上。
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(),requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    net_bc_out = net(pt_x_bc, pt_t_bc)  # 将边界条件下的 x 和 t 输入到神经网络中，得到网络的输出。
   #使用均方误差损失函数计算网络输出与真实值 pt_u_bc 之间的损失，这个损失反映了网络在边界条件上的表现。
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)

    #基于偏微分方程的损失计算
    x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500, 1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500, 1))
    all_zeros = np.zeros((500, 1))

    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    f_out = f(pt_x_collocation, pt_t_collocation, net)  # output of f(x,t)
    mse_f = mse_cost_function(f_out, pt_all_zeros)

    ##将基于边界条件的损失和基于偏微分方程的损失相加，得到总的损失。
    loss = mse_u + mse_f
    lo.append(loss.detach().numpy())
    ite.append(epoch)
    loss.backward()##进行反向传播，计算损失关于网络参数的梯度
    optimizer.step()##使用优化器根据计算得到的梯度更新网络参数。
#    with torch.autograd.no_grad():##with是一个关键字，用于创建一个上下文管理器
  #      print(epoch, "Traning Loss:", loss.data)
##绘图
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
fig=plt.figure(1)
ax=plt.axes(projection='3d')
x=np.arange(0,2,0.02)##生成一个从 0 到 2，步长为 0.02 的一维数组
t=np.arange(0,1,0.02)
ms_x,ms_t=np.meshgrid(x,t)#使用 numpy 的 meshgrid 函数将 x 和 t 转换为二维网格坐标。
##将二维网格坐标展平为一维数组，然后再重新调整形状为列向量。
x = np.ravel(ms_x).reshape(-1,1)
t = np.ravel(ms_t).reshape(-1,1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True).to(device)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True).to(device)
pt_u = net(pt_x,pt_t)##使用训练好的神经网络 net 对 pt_x 和 pt_t 进行计算，得到网络的输出。
u=pt_u.data.cpu().numpy()##将 PyTorch 的张量转换为 numpy 数组，并将其移动到 CPU 上
ms_u = u.reshape(ms_x.shape)##将一维的 u 数组重新调整形状为与 ms_x 相同的二维形状

surf=ax.plot_surface(ms_x,ms_t,ms_u,cmap=cm.coolwarm,linewidth=0,antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10)) #设置 z 轴的刻度定位器和格式，这里设置为有 10 个主要刻度
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))#格式设置为保留两位小数
fig.colorbar(surf, shrink=0.5, aspect=5)#shrink=0.5 表示颜色条的大小为图形的 0.5 倍，aspect=5 表示颜色条的长宽比为 5。
plt.figure(2)
plt.plot(ite,lo)
end_time=time.time()
times=end_time-start_time

#torch.save(net.state_dict(), "model_uxt.pt")


print(f"运行时间为：{times}秒")
plt.show()