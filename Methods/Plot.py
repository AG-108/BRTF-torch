import torch
import matplotlib.pyplot as plt

def PlotYXS(*args):
    nTensor = len(args)
    strName = ['Observed Tensor','Low Rank','Sparse','Noise']
    parThresh = torch.zeros(1,nTensor)
    parThresh[3] = 0.2

    for i in range(nTensor):
        plt.subplot(1,4,i)


import matplotlib.pyplot as plt
import numpy as np

def tensorHeatmap(data, name="image"):
    array = data.numpy()

    # 获取张量的下标和值
    index_x, index_y, index_z = np.meshgrid(np.arange(data.size(dim=0)),np.arange(data.size(dim=1)),np.arange(data.size(dim=2)))

    # 绘制三维散点图
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(index_x, index_y, index_z, c=array, cmap="plasma")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.savefig(name)
    plt.show()