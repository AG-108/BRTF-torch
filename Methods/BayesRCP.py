import tltorch
from .config import *
import torch
import numpy as np
import tensorly as tl

def ten2mat(tensor, mode):
    return torch.tensor(
        np.reshape(
            np.moveaxis(tensor.numpy(), mode, 0),
            (tensor.shape[mode], -1),
            order='F')
    )

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

    dimY = torch.tensor(Y.size())
    N = dimY.size(dim=0)
    nObs = dimY.prod()
    LB = 0

    a_gamma0 = 1e-6
    b_gamma0 = 1e-6
    a_beta0 = 1e-6
    b_beta0 = 1e-6
    a_alpha0 = 1e-6
    b_alpha0 = 1e-6
    eps = 1e-15

    gammas = (a_gamma0+eps)/(b_gamma0+eps)*torch.ones(maxRank,1);
    beta = (a_beta0+eps)/(b_beta0+eps);
    alphas = 1/initVar*torch.ones(tuple(dimY))*((a_alpha0+eps)/(b_alpha0+eps));

    E = 1/alphas.sqrt()*torch.randn(tuple(dimY));
    Sigma_E = 1/alphas*torch.ones(tuple(dimY));

    if not isinstance(init,int):
        pass
        # Z = init;
        # if numel(Z) ~= N
        #     error('OPTS.init does not have %d cells',N);
        # end
        # for n = 1:N;
        #     if ~isequal(size(Z{n}),[size(Y,n) maxRank])
        #         error('OPTS.init{%d} is the wrong size',n);
        #     end
        # end
    else:
        if init == INIT_ML:
            Z = []
            ZSigma = []
            for n in range(N):
                ZSigma.append((1/gammas).diag())
                [U, S, Vh] = torch.linalg.svd(ten2mat(Y, n))
                S = S.diag()
                V = Vh.mH
                dvar = (Y.square().sum()/nObs).sqrt()
                if maxRank <= U.size(dim=1):
                    Z.append(U[:,0:maxRank].matmul(S[0:maxRank,0:maxRank].sqrt()))
                else:
                    Z.append(torch.cat(U.matmul(S.sqrt()), torch.randn(dimY[n], maxRank-U.size(dim=1))*dvar))
        elif init == INIT_RAND:
            Z = []
            ZSigma = []
            for n in range(n):
                ZSigma.append((1/gammas).diag())
                dvar = (Y.square().sum() / nObs).sqrt()
                Z.append(torch.randn(dimY[n], maxRank) * dvar)

    # create figure -- remain to be solved

    # model learning
    EZZT = []
    for n in range(N):
        EZZT.append(Z[n].mH.matmul(Z[n]) + dimY[n]*ZSigma[n])

    for it in range(maxIters):
        Aw = gammas.diag()
        for n in range(N):
            ENZZT = torch.ones(maxRank,maxRank)
            for m in range(N):
                if m != n:
                    ENZZT = ENZZT * EZZT[m]
            FslashY = tl.tenalg.khatri_rao(Z[0:n]+Z[n+1:N], reverse=True).mH.matmul(ten2mat(Y-E,n).mH)
            ZSigma[n] = 1/(beta * ENZZT + Aw)
            Z[n] = (beta * ZSigma[n]).matmul(FslashY)
            EZZT[n] = Z[n].mH.matmul(Z[n]) + dimY[n] * ZSigma[n]

    print(len(Z))
    X = tltorch.CPTensor(torch.ones(Z[0].size(dim=1)),Z).to_tensor()

    a_gammaN = (0.5*dimY.sum() + a_gamma0) * torch.ones(maxRank,1)
    b_gammaN = 0
    for n in range(N):
        b_gammaN = b_gammaN + (Z[n].mH.matmul(Z[n])).diag() + dimY[n] * ZSigma[n].diag
    b_gammaN = b_gamma0 + 0.5 * b_gammaN
    gammas = a_gammaN/b_gammaN

    EX2 = torch.ones(maxRank,maxRank)
    for n in range(N):
        EX2 = EX2 * EZZT[n]
    EX2 = EX2.sum(dim=0)
    return None

