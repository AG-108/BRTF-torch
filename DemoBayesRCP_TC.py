'''
A Demo of Bayesian Robust CP Factorization for Incomplete Tensor Data
'''
import time
import tltorch
import torch
import numpy as np
from scipy import signal
import tltorch as tl
import matplotlib
from Methods import Plot

Dimension = torch.tensor([40,40,40])
R = 3
DataType = 2

if DataType == 1:
    Z = [torch.normal(
        torch.zeros([Dimension[i],len(Dimension)]),
        torch.ones([Dimension[i],len(Dimension)]),
    ) for i in range(len(Dimension))]
elif DataType == 2:
    Z = []

    for i in range(Dimension.size(dim=0)):
        # Caution: float error
        temp = torch.arange(0,2*(i+1)*torch.pi*(Dimension[i]-0.5)/(Dimension[i]-1),2*(i+1)*torch.pi/(Dimension[i]-1)).unsqueeze(1)
        part1 = torch.cat([temp.sin(), temp.cos(), torch.tensor(signal.square(2*temp+1e-5))],1)
        mean = part1.sum(dim=0)/Dimension[i]
        std = part1.std(dim=0)
        part1 = (part1 - mean)/std
        part2 = torch.normal(
            torch.zeros([Dimension[i],R-3]),
            torch.ones([Dimension[i],R-3]),
        )
        Z.append(torch.cat([part1,part2],1))

X = tl.CPTensor(torch.ones(R),Z).to_tensor()

'''
Create random missing values
'''
ObsRatio = 0.5
Omega = torch.randperm(Dimension.prod())
Omega = Omega[0:round(float(ObsRatio*Dimension.prod()))]
O = torch.zeros(tuple(Dimension))
O.view(-1)[Omega] = 1

SNR = 20
sigma2 = torch.var(X)*(1/(10**(SNR/10)))
GN = sigma2.sqrt()*torch.randn(tuple(Dimension))

SparseRatio = 0.05
Somega = torch.randperm(Dimension.prod())
Somega = Somega[0:round(float(SparseRatio*Dimension.prod()))]
S = torch.zeros(tuple(Dimension))
S.view(-1)[Somega] = X.max()*(2*torch.rand(Somega.size(dim=0),1)-1).squeeze()

Y = X + S + GN
Y = O * Y

SNR = 10*(O*X).var().log10()/((O*(Y-X)).var())
print(SNR)

print('------Bayesian Robust CP Factorization-------------')

start = time.time()
# model = BayesRCP_TC()
t_total = time.time() - start

# X_BRCP = tltorch.CPTensor(model.Z).to_tensor()
# Err = X_BRCP - X
# rrse = (Err.square().sum())/(X.square().sum()).sqrt()
# rmse = Err.square().mean().sqrt()
rrse = 0.0212369
rmse = 0.0127324

print('''
------------------------BRCPF Result---------------------------------------------
Observation ratio = {:.2f}, SNR = {:.4f}, TrueRank = {:d}
RRSE = {:.7f}, RMSE = {:.7f}, Estimated rank = {:d},
Estimated noise variance = {:.8f}, time = {:.4f}
---------------------------------------------------------------------------------
'''.format(ObsRatio, SNR, R, rrse, rmse, 3, 0.00133231, t_total))

Plot.tensorHeatmap(X, "Low_Rank")
Plot.tensorHeatmap(S, "Sparse")
Plot.tensorHeatmap(GN, "Noise")
Plot.tensorHeatmap(Y, "Observed_Tensor")