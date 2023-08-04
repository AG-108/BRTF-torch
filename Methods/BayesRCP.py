import scipy.optimize
import tltorch
from .config import *
import torch
import numpy as np
import tensorly as tl
import random
import math

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# def ten2mat(tensor, mode):
#     return torch.tensor(
#         np.reshape(
#             np.moveaxis(tensor.numpy(), mode, 0),
#             (tensor.shape[mode], -1),
#             order='F')
#     )

def tenmat_permute_idx(n, N):
    permute_arrange = list(range(N - 1, -1, -1))
    permute_arrange.remove(n)
    return [n] + permute_arrange

def ten2mat(X, n, N):
    idx = tenmat_permute_idx(n, N)
    return X.permute(idx).flatten(1, -1)

def safelog(x):
    return math.log(min(max(x, 1e-300), 1e300)) if isinstance(x, (float, int)) \
        else torch.log(torch.clamp(x, min=1e-45, max=1e38))
    # if not isinstance(x,torch.Tensor):
    #     x = torch.tensor(x)
    # x = torch.clamp(x,1e-38,1e38)
    # return torch.log(x)

def BayesRCP(Y, init = INIT_ML, maxRank = None, dimRed = 1, initVar = 1, updateHyper = UPDATEHYPER_ON, maxIters = 100, tol = 1e-5, predVar = PREDVAR_DOES_NOT_COMPUTE, verbose = VERBOSE_TEXT):
    '''
    :param Y:               - input tensor
    :param init:            - Initialization method
                                - 'ml'  : Apply SVD to Y and initialize factor matrices (default)
                                - 'rand': initialize factor matrices with random matrices
    :param maxRank:         - The initial CP rank
                            - max(size(Y)) (default)
    :param dimRed:          - 0: Keep number of components as the initialized value
                            - 1: Remove zero components automatically (default)
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

    a_gamma0 = torch.tensor(1e-6)
    b_gamma0 = torch.tensor(1e-6)
    a_beta0 = torch.tensor(1e-6)
    b_beta0 = torch.tensor(1e-6)
    a_alpha0 = torch.tensor(1e-6)
    b_alpha0 = torch.tensor(1e-6)
    eps = 2.2204e-16
    # eps = torch.tensor(1e-10)

    gammas = (a_gamma0+eps)/(b_gamma0+eps)*torch.ones(maxRank)#,1);
    beta = (a_beta0+eps)/(b_beta0+eps);
    alphas = 1/initVar*torch.ones(tuple(dimY))*((a_alpha0+eps)/(b_alpha0+eps));

    # E = 1/(alphas.sqrt())*torch.randn(tuple(dimY));
    # Sigma_E = 1/alphas*torch.ones(tuple(dimY));
    E = torch.pow(alphas, -0.5) * torch.randn_like(Y)
    Sigma_E = torch.pow(alphas, -1) * torch.ones_like(Y)

    # if not isinstance(init,int):
    #     pass
    #     # Z = init;
    #     # if numel(Z) ~= N
    #     #     error('OPTS.init does not have %d cells',N);
    #     # end
    #     # for n = 1:N;
    #     #     if ~isequal(size(Z{n}),[size(Y,n) maxRank])
    #     #         error('OPTS.init{%d} is the wrong size',n);
    #     #     end
    #     # end
    # else:
    #     dvar = (Y.square().sum() / nObs).sqrt()
    #     if init == INIT_ML:
    #         Z = []
    #         ZSigma = []
    #         for n in range(N):
    #             ZSigma.append((1/gammas).diag())
    #             [U, S, Vh] = torch.linalg.svd(ten2mat(Y, n, N), full_matrices=False)
    #             S = S.diag()
    #             V = Vh.mH
    #             if maxRank <= U.size(dim=1):
    #                 Z.append(U[:,0:maxRank].matmul(S[0:maxRank,0:maxRank].sqrt()))
    #             else:
    #                 Z.append(torch.cat(U.matmul(S.sqrt()), torch.randn(dimY[n], maxRank-U.size(dim=1))*dvar))
    #     elif init == INIT_RAND:
    #         Z = []
    #         ZSigma = []
    #         for n in range(N):
    #             ZSigma.append((1/gammas).diag())
    #             Z.append(torch.randn(dimY[n], maxRank) * dvar)
    Z, ZSigma = [], []
    dvar = torch.sqrt(torch.sum(torch.square(Y)) / nObs)
    for n in range(N):
        ZSigma.append(torch.diag(torch.pow(gammas, -1)))
        U, S, V = torch.linalg.svd(ten2mat(Y, n, N).flatten(1, -1), full_matrices=False)
        S = torch.diag(S)
        if maxRank <= U.size(1):
            Z.append(U[:, :maxRank] @ torch.pow(S[:maxRank, :maxRank], 0.5))
        else:
            Z.append(
                torch.concat([U @ torch.pow(S, 0.5), torch.randn((dimY[n], maxRank - U.size(1))).to(Y.device) * dvar],
                                 dim=1))
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
            FslashY = tl.tenalg.khatri_rao(Z[0:n]+Z[n+1:N], reverse=True).mH.matmul(ten2mat(Y-E,n, N).mH)
            ZSigma[n] = torch.inverse(beta * ENZZT + Aw)
            Z[n] = (beta * ZSigma[n]).matmul(FslashY).mH
            EZZT[n] = Z[n].mH.matmul(Z[n]) + dimY[n] * ZSigma[n]

        X = tltorch.CPTensor(torch.ones(Z[0].size(dim=1)),Z).to_tensor()

        a_gammaN = (0.5*dimY.sum() + a_gamma0) * torch.ones(maxRank)#,1)
        b_gammaN = 0
        for n in range(N):
            b_gammaN = b_gammaN + (Z[n].mH.matmul(Z[n])).diag() + dimY[n] * ZSigma[n].diag()
        b_gammaN = b_gamma0 + 0.5 * b_gammaN
        gammas = a_gammaN/b_gammaN

        EX2 = torch.ones(maxRank,maxRank)
        for n in range(N):
            EX2 = EX2 * EZZT[n]
        EX2 = EX2.sum()
        EE2 = (E.square() + Sigma_E).sum()
        Err = Y.square().sum() \
              - 2*(Y.view(-1)*X.view(-1)).sum() \
              - 2*(Y.view(-1)*E.view(-1)).sum() \
              + 2*(E.view(-1)*X.view(-1)).sum() + EX2 + EE2
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
        a_alphaN = torch.tensor(a_alphaN)
        a_alpha0 = torch.tensor(a_alpha0)
        items = torch.zeros(11)

        items[0] = -0.5 * nObs * safelog(torch.tensor(2 * torch.pi)) \
                + 0.5 * nObs * (torch.special.psi(a_betaN) - safelog(b_betaN)) \
                - 0.5 * (a_betaN / b_betaN) * Err

        items[1] = 0
        #
        for n in range(N):
            items[1] = items[1] + -0.5 * maxRank * dimY[n] * safelog(torch.tensor(2*torch.pi)) \
                    + 0.5 * dimY[n] * (torch.special.psi(a_gammaN)- safelog(b_gammaN)).sum() \
                    - 0.5 * (gammas.diag().matmul(ZSigma[n])).trace() \
                    - 0.5 * (gammas.diag() * Z[n].mH.matmul(Z[n])).trace()
            # 这里有一个求和：为什么？

        items[2] = (-torch.lgamma(a_gamma0)
                    + a_gamma0 * safelog(b_gamma0) \
                    - b_gamma0 * (a_gammaN / b_gammaN) \
                    + (a_gamma0 - 1) * (torch.special.psi(a_gammaN) - safelog(b_gammaN))).sum()

        items[3] = -torch.lgamma(a_beta0) + a_beta0 * safelog(b_beta0) \
                + (a_beta0 - 1) * (torch.special.psi(a_betaN) - safelog(b_betaN)) \
                - b_beta0 * (a_betaN / b_betaN);

        items[4] = 0.5 * maxRank * (dimY.sum()) * (1 + safelog(2*torch.pi));

        for n in range(N):
            items[4] = items[4] + dimY[n] * 0.5 * safelog((ZSigma[n]).det())

        items[5] = (torch.lgamma(a_gammaN) \
                    - (a_gammaN - 1) * torch.special.psi(a_gammaN) - safelog(b_gammaN) + a_gammaN).sum();

        items[6] = torch.lgamma(a_betaN) - (a_betaN - 1) * torch.special.psi(a_betaN) \
                   - safelog(b_betaN) + a_betaN;

        temp = torch.special.psi(a_alphaN) - safelog(b_alphaN);

        items[7] = - 0.5 * nObs * safelog(2*torch.pi) + 0.5 * temp.sum() \
                - 0.5 * ((E.square() + Sigma_E)*alphas).sum()

        items[8] = -nObs * torch.lgamma(a_alpha0) + nObs * a_alpha0 * safelog(b_alpha0) \
                + ((a_alpha0-1) * temp - b_alpha0 * alphas).sum()

        items[9] = 0.5 * (safelog(Sigma_E)).sum() + 0.5 * nObs * (1 + safelog(2 * torch.pi))

        items[10] = (torch.lgamma(a_alphaN) \
                 - (a_alphaN - 1) * torch.special.psi(a_alphaN) \
                 - safelog(b_alphaN) + a_alphaN).sum()

        LB.append(items.sum())

        if updateHyper == UPDATEHYPER_ON:
            if it > 5:
                aMean = alphas.mean().item()
                bMean = (torch.digamma(torch.tensor(a_alphaN)) - safelog(b_alphaN.flatten())).mean().item()
                ngLB = lambda x: math.lgamma(x) - x * math.log(x / aMean) - (x - 1) * bMean + x
                a_alpha0 = scipy.optimize.fmin(ngLB, a_alpha0, disp=False)[0]
                # aMean = alphas.mean()
                # bMean = (torch.special.psi(a_alphaN) - safelog(b_alphaN)).mean()
                # ngLB = lambda x : np.math.lgamma(x) - x * np.log(x/aMean.numpy()) - (x-1)*(bMean.numpy()) + x
                # a_alpha0 = scipy.optimize.minimize(ngLB,a_alpha0.numpy()).x
                # a_alpha0 = torch.tensor(a_alpha0,dtype=torch.float32)
                b_alpha0 = a_alpha0/aMean

        Zall = torch.concat(Z,dim=0)
        comPower = (Zall.mH.matmul(Zall)).diag()
        comTol = dimY.sum() * np.finfo(float(Zall.norm())).eps
        rankest = (comPower > comTol).sum()

        if rankest.max() == 0:
            raise ValueError("Rank becomes 0!")

        if dimRed == 1 and it >= 2:#1:
            if maxRank != rankest.max():
                indices = comPower > comTol
                gammas = gammas[indices]
                for n in range(N):
                    Z[n] = Z[n][:,indices]
                    # ZSigma[n] = ZSigma[n][indices,indices]
                    # EZZT[n] = EZZT[n][indices,indices]
                    ZSigma[n] = ZSigma[n][indices, :][:, indices]
                    EZZT[n] = EZZT[n][indices, :][:, indices]

                maxRank = rankest

        # visualize online results

        # Display progress

        # Convergence check

    # Predictive distribution
    if predVar == PREDVAR_COMPUTE_AND_OUTPUT:
        XVar = torch.zeros(Y.size())
        for i in range(N):
            XVar = ten2mat(XVar,dim=n)
            Fslash = tl.tenalg.khatri_rao(Z[0:n]+Z[n+1:N], reverse=True)
            XVar = XVar + (Fslash.matmul(ZSigma[n]).matmul(Fslash.mH)).mH.repeat(dimY[n],1)
        XVar = XVar + 1/beta
        XVar = XVar*(2*a_betaN)/(2*a_betaN-2)
    elif predVar == PREDVAR_FAST_COMPUTATION:
        temp = []
        for i in range(N):
            temp.append((ZSigma[n].repeat(1,dimY[n]) + tl.tenalg.khatri_rao(Z[n].mH,Z[n].mH)).mH)
        XVar = tltorch.CPTensor(torch.ones(n),temp).to_tensor() - X.square()
        XVar = XVar + 1/beta
    else:
        XVar = torch.tensor([])

    model = {}
    # output
    model['Z'] = Z
    model['ZSigma'] = ZSigma
    model['gammas'] = gammas
    model['E'] = E
    model['Sigma_E'] = Sigma_E
    model['beta'] = beta
    model['XVar'] = XVar
    model['TrueRank'] = rankest
    model['LowBound'] = max(LB)

    return model

