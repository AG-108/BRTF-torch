from config import *
import torch

def ten2mat(tensor, mode):
    return tensor.moveaxis(mode, 0).reshape((tensor.shape[mode], -1))

def BayesRCP(Y, init = INIT_ML, maxRank = None, dimRed = 1, initVar = 1, updateHyper = UPDATEHYPER_ON, maxIters = 100, tol = 1e-5, predVar = PREDVAR_DOES_NOT_COMPUTE, verbose = VERBOSE_TEXT):
    '''
    :param Y:               - input tensor
    :param init:            - Initialization mathod
                                - 'ml'  : Apply SVD to Y and initialize factor matrices (default)
                                - 'rand': initialize factor matrices with random matrices
    :param maxRank:         - The initial CP rank
                            - max(size(Y)) (default)
    :param dimRed:          - 0: Keep number of components as the initialized value
                            - 1: Remove zero components automaticly (default)
    :param initVar:         - Initialization of variance of outliers (default: 1)
    :param updateHyper:     - Optimization of top level parameter (default: on)
    :param maxIters:        - maximum number of iterations (default: 100)
    :param tol:             - lower band change tolerance for convergence dection (default: 1e-5)
    :param predVar:         - Predictive confidence (default: does not compute)
    :param verbose:         - visualization of results (default: text)

    :return: model          - Model parameters and hyperparameters
    '''
    if maxRank == None:
        maxRank = Y.size().max()

    # Initialization

    dimY = Y.size();
    N = dimY.size(dim=0);
    nObs = prod(dimY)
    LB = 0

    a_gamma0 = 1e-6
    b_gamma0 = 1e-6
    a_beta0 = 1e-6
    b_beta0 = 1e-6
    a_alpha0 = 1e-6
    b_alpha0 = 1e-6
    eps = 1e-15

    gammas = (a_gamma0+eps)/(b_gamma0+eps)*torch.ones(R,1);
    beta = (a_beta0+eps)/(b_beta0+eps);
    alphas = 1/initVar*torch.ones(dimY)*((a_alpha0+eps)/(b_alpha0+eps));

    E = 1/alphas.sqrt()*torch.randn(dimY);
    Sigma_E = 1/alphas*torch.ones(dimY);

    if not isinstance(init,int):
        pass
        # Z = init;
        # if numel(Z) ~= N
        #     error('OPTS.init does not have %d cells',N);
        # end
        # for n = 1:N;
        #     if ~isequal(size(Z{n}),[size(Y,n) R])
        #         error('OPTS.init{%d} is the wrong size',n);
        #     end
        # end
    else:
        if init == INIT_ML:
            Z = []
            ZSigma = []
            for n in range(n):
                ZSigma[n] = (1/gammas).diag()

X = torch.tensor([[[1, 2, 3, 4], [3, 4, 5, 6]],
              [[5, 6, 7, 8], [7, 8, 9, 10]],
              [[9, 10, 11, 12], [11, 12, 13, 14]]])

print('tensor size:')
print(X.shape)
print()
print('切片矩阵：X[:, :, 1] =')
print(X[:, :, 0])
print()
print('切片矩阵：X[:, :, 2] =')
print(X[:, :, 1])
print()
print('切片矩阵：X[:, :, 3] =')
print(X[:, :, 2])
print()
print('切片矩阵：X[:, :, 4] =')
print(X[:, :, 3])
print()
print('模态1展开矩阵 (mode-1 tensor unfolding):')
print(ten2mat(X, 0))