'''
A Demo of Bayesian Robust CP Factorization for Incomplete Tensor Data
'''
import time
from scipy import signal
from Methods.BayesRCP import *

# Generate a low-rank tensor
# Dimensions
Dimension = torch.tensor([30, 30, 30])  # ture rank for generating the tensor data
R = 5
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
        temp = torch.arange(0, 2*(i+1)*torch.pi*(Dimension[i]-0.5)/(Dimension[i]-1), 2*(i+1)*torch.pi/(Dimension[i]-1)).unsqueeze(1)
        part1 = torch.cat([temp.sin(), temp.cos(), torch.tensor(signal.square(2*temp+1e-5), dtype=torch.float32)], 1)
        part2 = torch.normal(
            torch.zeros([Dimension[i], R-3]),
            torch.ones([Dimension[i], R-3]),
        )
        Z.append(torch.cat([part1, part2], 1))

X = tltorch.CPTensor(torch.ones(R), Z).to_tensor()

SNR = 20
sigma2 = X.var()*(1/(10**(SNR/10)))
GN = sigma2.sqrt()*torch.randn(tuple(Dimension))

SparseRatio = 0.05
Somega = torch.randperm(Dimension.prod())
Somega = Somega[0:round(float(SparseRatio*Dimension.prod()))]
S = torch.zeros(tuple(Dimension))
S.view(-1)[Somega] = X.max()*(2*torch.rand(Somega.size(dim=0), 1)-1).squeeze()

Y = X + S + GN

SNR = 10*((X.var()/(Y-X).var()).log10())

print('------Bayesian Robust CP Factorization-------------')

start = time.time()
model = BayesRCP(Y, maxRank=max(list(Dimension)+[2*R]), verbose=VERBOSE_IMAGE_PLOT)
t_total = time.time() - start

X_BRCP = tltorch.CPTensor(model.Z).to_tensor()
Err = X_BRCP - X
rrse = (Err.square().sum())/(X.square().sum()).sqrt()
rmse = Err.square().mean().sqrt()

print('''
------------------------BRCPF Result---------------------------------------------
SNR = {:.4f}, TrueRank = {:d}
RRSE = {:.7f}, RMSE = {:.7f}, Estimated rank = {:d},
Estimated noise variance = {:.8f}, time = {:.4f}
---------------------------------------------------------------------------------
'''.format(SNR, R, rrse, rmse, 3, 0.00133231, t_total))

'''
对照组：使用Tensorly.decomposition中的parafac方法，其就使用了CP-ALS法来计算张量的CP分解
'''

from tensorly.decomposition import parafac

# 进行CP分解，设置rank为2，return_errors为True
factors, errors = parafac(Y, rank=R, return_errors=True)

# 打印误差列表
for i in errors:
    print(float(i))

# rrse = (Err.square().sum())/(X.square().sum()).sqrt()
# rmse = Err.square().mean().sqrt()

print('''-------------CP-ALS------------------------------------------
RRSE = {:.7f}, RMSE = {:.7f},
-------------------------------------------------------------'''.format(rrse, rmse))