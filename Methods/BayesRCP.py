import scipy.optimize
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

def safelog(x):
    x = max(x,1e-200)
    x = min(x,1e300)
    return torch.log(x)

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
    LB = []

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

        X = tltorch.CPTensor(torch.ones(Z[0].size(dim=1)),Z).to_tensor()

        a_gammaN = (0.5*dimY.sum() + a_gamma0) * torch.ones(maxRank,1)
        b_gammaN = 0
        for n in range(N):
            b_gammaN = b_gammaN + (Z[n].mH.matmul(Z[n])).diag() + dimY[n] * ZSigma[n].diag()
        b_gammaN = b_gamma0 + 0.5 * b_gammaN
        gammas = a_gammaN/b_gammaN

        EX2 = torch.ones(maxRank,maxRank)
        for n in range(N):
            EX2 = EX2 * EZZT[n]
        EX2 = EX2.sum(dim=0)
        EE2 = (E.square() + Sigma_E).sum()
        Err = Y.mH.matmul(Y) - Y.mH.matmul(E) + 2 * X.mH.matmul(E) + EX2 + EE2
        a_betaN = a_beta0 + 0.5*nObs
        b_betaN = b_beta0 + 0.5*Err
        beta = a_betaN/b_betaN

        Sigma_E = 1/(alphas + beta)
        E = beta * (Y-X) * Sigma_E

        inf_flag = 1
        if inf_flag == 1:
            a_alphaN = a_alpha0 + 0.5
            b_alphaN = b_alpha0 + 0.5 * (E.square() + Sigma_E)
            alphas = a_alphaN/b_alphaN
        elif inf_flag == 2:
            a_alphaN = a_alpha0 + 1 - alphas * Sigma_E
            b_alphaN = b_alpha0 + E.square() + eps
            alphas = a_alphaN / b_alphaN

        items = torch.zeros(11)

        items[1] = -0.5 * nObs * safelog(2 * torch.pi) \
                + 0.5 * nObs * (psi(a_betaN) - safelog(b_betaN)) \
                - 0.5 * (a_betaN / b_betaN) * err;

        items[2] = 0

        for n in range(N):
            items[2] = items[2] + -0.5 * maxRank * dimY[n] * safelog(2*pi) \
                    + 0.5 * dimY[n] * (torch.special.psi(a_gammaN)- safelog(b_gammaN)).sum() \
                    - 0.5 * (gammas.diag() * (ZSigma[n].sum(dim=2))).trace() \
                    - 0.5 * (gammas.diag() * Z[n].mH.matmul(Z[n])).trace()

        items[3] = (-safelog(torch.lgamma(a_gamma0)).sum() + a_gamma0 * safelog(b_gamma0) \
                - b_gamma0 * (a_gammaN / b_gammaN) \
                + (a_gamma0 - 1) * (torch.special.psi(a_gammaN) - safelog(b_gammaN)))

        items[4] = -safelog(torch.lgamma(a_beta0)) + a_beta0 * safelog(b_beta0) \
                + (a_beta0 - 1) * (torch.special.psi(a_betaN) - safelog(b_betaN)) \
                - b_beta0 * (a_betaN / b_betaN);

        items[5] = 0.5 * maxRank * dimY.sum() * (1 + safelog(2*pi));

        for n in range(N):
            items[5] = items[5] + dimY[n] * 0.5 * safelog((ZSigma[n]).det())

        items[6] = sum(safelog(gamma(a_gammaN)) - (a_gammaN - 1) * psi(a_gammaN) - safelog(b_gammaN) + a_gammaN);

        items[7] = safelog(gamma(a_betaN)) - (a_betaN - 1) * psi(a_betaN) - safelog(b_betaN) + a_betaN;

        temp = torch.special.psi(a_alphaN) - safelog(b_alphaN);

        items[8] = -0.5 * nObs * safelog(2*pi) + 0.5 * temp.sum() \
                - 0.5 * (E.square() + Sigma_E).mH.matmul(alphas)

        items[9] = -nObs * safelog(gamma(a_alpha0)) + nObs * a_alpha0 * safelog(b_alpha0) \
                + ((a_alpha0-1) * temp - b_alpha0 * alphas).sum()

        items[10] = 0.5 * (safelog(Sigma_E)),sum() + 0.5 * nObs * (1 + safelog(2 * pi))

        items[11] = (safelog(torch.lgamma(a_alphaN)) \
                 - (a_alphaN - 1) * torch.special.psi(a_alphaN) \
                 - safelog(b_alphaN) + a_alphaN).sum()

        LB.append(items.sum())

        if updateHyper == UPDATEHYPER_ON:
            if it > 5:
                aMean = alpha.mean()
                bMean = (torch.special.psi(a_alphaN) - safelog(b_alphaN)).mean()
                ngLB = lambda x : x.lgamma().log() - x * (x/aMean).log() - (x-1)*bMean + x
                a_alpha0 = scipy.optimize.minimize(ngLB,a_alpha0)
                b_alpha0 = a_alpha0/aMean

    
    return None

