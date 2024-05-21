import sys
import os
import numpy as np
import torch
import torch.nn.functional as F


def k2i_torch(K, ax=[-3, -2, -1]):
    X = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(K, dim=ax), dim=ax, norm="ortho"),
                           dim=ax)
    return X


def i2k_torch(K, ax=[-3, -2, -1]):
    X = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(K, dim=ax), dim=ax, norm="ortho"),
                           dim=ax)
    return X


class Eop():
    def __init__(self):
        super(Eop, self).__init__()

    def mtimes(self, b, inv, sens, us_mask):
        if inv:
            # b: nv,nt,nc,x,y,z
            x = torch.sum(k2i_torch(b * us_mask, ax=[-3, -2, -1]) * torch.conj(sens), dim=2)
        else:
            b = b.unsqueeze(2) * sens
            x = i2k_torch(b, ax=[-3, -2, -1]) * us_mask
        return x


def SoftThres(X, reg):
    X = torch.sgn(X) * (torch.abs(X) - reg) * ((torch.abs(X) - reg) > 0)
    return X


def SVT(X, reg):
    Np, Nt, FE, PE, SPE = X.shape
    reg *= (np.sqrt(np.prod(X.shape[-3:])) + 1)
    U, S, Vh = torch.linalg.svd(X.view(Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    X = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Np, Nt, FE, PE, SPE)
    return X, torch.sum(S_new)


def SVT_CP(X, reg):
    Np, Nt, FE, PE, SPE = X.shape
    reg *= (np.sqrt(np.prod(X.shape[-3:])) + 1)
    U, S, Vh = torch.linalg.svd(X.view(Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    X = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Np, Nt, FE, PE, SPE)
    return X, torch.sum(S_new)


def SVT_LLR(X, reg, blk):
    def GETWIDTH(M, N, B):
        temp = (np.sqrt(M) + np.sqrt(N))
        if M > N:
            return temp + np.sqrt(np.log2(B * N))
        else:
            return temp + np.sqrt(np.log2(B * M))

    Np, Nt, FE, PE, SPE = X.shape
    stepx = np.ceil(FE / blk)
    stepy = np.ceil(PE / blk)
    stepz = np.ceil(SPE / blk)
    padx = (stepx * blk).astype('uint16')
    pady = (stepy * blk).astype('uint16')
    padz = (stepz * blk).astype('uint16')
    rrx = torch.randperm(blk)[0]
    rry = torch.randperm(blk)[0]
    rrz = torch.randperm(blk)[0]
    X = F.pad(X, (0, padz - SPE, 0, pady - PE, 0, padx - FE))
    X = torch.roll(X, (rrz, rry, rrx), (-1, -2, -3))
    FEp, PEp, SPEp = X.shape[-3:]
    patches = X.unfold(2, blk, blk).unfold(3, blk, blk).unfold(4, blk, blk)
    unfold_shape = patches.size()
    patches = patches.contiguous().view(Np, Nt, -1, blk, blk, blk).permute((2, 0, 1, 3, 4, 5))
    Nb = patches.shape[0]
    M = blk ** 3
    N = blk ** 3 / M
    B = FEp * PEp * SPEp / blk ** 3
    RF = GETWIDTH(M, N, B)
    reg *= RF
    U, S, Vh = torch.linalg.svd(patches.view(Nb, Np * Nt, -1), full_matrices=False)
    S_new = SoftThres(S, reg)
    S_new = torch.diag_embed(S_new).to(torch.complex64)
    patches = torch.linalg.matmul(torch.linalg.matmul(U, S_new), Vh).view(Nb, Np, Nt, blk, blk, blk)
    patches = patches.permute((1, 2, 0, 3, 4, 5))
    patches_orig = patches.view(unfold_shape)
    patches_orig = patches_orig.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous()
    patches_orig = patches_orig.view(Np, Nt, FEp, PEp, SPEp)
    patches_orig = torch.roll(patches_orig, (-rrz, -rry, -rrx), (-1, -2, -3))
    X = patches_orig[..., :FE, :PE, :SPE]
    return X, torch.sum(S_new)


def GLR_ISTA(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    for i in range(it):
        X, lnu = SVT(X, reg)
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        X = X - A.mtimes(axb, 1, csm, us_mask)
    return X


def GLR_FISTA(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    tp = 1
    Y = X.clone()
    Xp = X.clone()
    loss_list = np.zeros((it))

    def PL(X, Kv, csm, us_mask, reg):
        X, lnu = SVT(X, reg)
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        X = X - A.mtimes(axb, 1, csm, us_mask)
        l2 = torch.sum(torch.abs(axb) ** 2)
        lnu = torch.abs(lnu)
        return X, l2, lnu

    for i in range(it):
        X, l2, lnu = PL(Y, Kv, csm, us_mask, reg)
        t = (1 + np.sqrt(1 + 4 * tp ** 2)) / 2
        Y = X + (tp - 1) / t * (X - Xp)
        tp = t
        Xp = X
    return X


def GLR_POGM(A, Kv, csm, us_mask, reg, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    t = 1
    Xp, Wp, W, Zp, Z = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
    thetap = 1
    gamma = 1
    for i in range(it):
        if i == it - 1:
            theta = 1 / 2 * (1 + np.sqrt(8 * thetap ** 2 + 1))
        else:
            theta = 1 / 2 * (1 + np.sqrt(4 * thetap ** 2 + 1))
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        W = Xp - t * A.mtimes(axb, 1, csm, us_mask)
        Z = W + (thetap - 1) / theta * (W - Wp) + thetap / theta * (W - Xp) + t / gamma * (thetap - 1) / theta * (
                Zp - Xp)
        gamma = t * (2 * thetap + theta - 1) / theta
        X, lnu = SVT(Z, reg * gamma)
        Xp = X
        Wp = W
        Zp = Z
        thetap = theta
    return X


def LLR_POGM(A, Kv, csm, us_mask, reg, it, blk):
    X = A.mtimes(Kv, 1, csm, us_mask)
    t = 1
    Xp, Wp, W, Zp, Z = X.clone(), X.clone(), X.clone(), X.clone(), X.clone()
    thetap = 1
    gamma = 1
    for i in range(it):
        if i == it - 1:
            theta = 1 / 2 * (1 + np.sqrt(8 * thetap ** 2 + 1))
        else:
            theta = 1 / 2 * (1 + np.sqrt(4 * thetap ** 2 + 1))
        axb = A.mtimes(X, 0, csm, us_mask) - Kv
        W = Xp - t * A.mtimes(axb, 1, csm, us_mask)
        Z = W + (thetap - 1) / theta * (W - Wp) + thetap / theta * (W - Xp) + t / gamma * (thetap - 1) / theta * (
                Zp - Xp)
        gamma = t * (2 * thetap + theta - 1) / theta
        X, lnu = SVT_LLR(Z, reg * gamma, blk)
        Xp = X
        Wp = W
        Zp = Z
        thetap = theta
    return X


def LplusS(A, Kv, csm, us_mask, regl, regs, it):
    X = A.mtimes(Kv, 1, csm, us_mask)
    S, Lp, Sp = torch.zeros_like(X), X.clone(), torch.zeros_like(X)

    def Sparse(S, reg, ax=[0, 1]):
        temp = SoftThres(i2k_torch(S, ax=ax), reg)
        return k2i_torch(temp, ax=ax), torch.sum(torch.abs(temp))

    for i in range(it):
        L, lnu = SVT(X - Sp, regl)
        S, ls = Sparse(X - Lp, regs, ax=[0, 1])
        axb = A.mtimes(L + S, 0, csm, us_mask) - Kv
        X = L + S - A.mtimes(axb, 1, csm, us_mask)
        Lp = L
        Sp = S
    return X, L, S


def LplusS_POGM(A, Kv, csm, us_mask, regl, regs, it):
    regl *= (np.sqrt(np.prod(Kv.shape[-3:])) + 1)
    M = A.mtimes(Kv, 1, csm, us_mask)
    X = torch.concat([M.clone().unsqueeze(0), torch.zeros_like(M).unsqueeze(0)], dim=0)
    X_, Xh, Xhp = X.clone(), X.clone(), X.clone()
    thetap, kesp = 1, 1
    t = 0.5

    def Sparse(S, reg, ax=[0, 1]):
        temp = SoftThres(i2k_torch(S, ax=ax), reg)
        return k2i_torch(temp, ax=ax), torch.sum(torch.abs(temp))

    for i in range(it):
        Xh[0] = M - X[1]
        Xh[1] = M - X[0]
        if i == it - 1:
            theta = (1 + np.sqrt(1 + 8 * thetap ** 2)) / 2
        else:
            theta = (1 + np.sqrt(1 + 4 * thetap ** 2)) / 2
        X_ = Xh + (thetap - 1) / theta * (Xh - Xhp) + thetap / theta * (Xh - X) \
             + (thetap - 1) / theta / kesp * t * (X_ - X)
        kes = t * (1 + (thetap - 1) / theta + thetap / theta)
        X[0], lnu = SVT(X_[0], regl)
        X[1], ls = Sparse(X_[1], regs, ax=[0, 1])
        axb = A.mtimes(X[0] + X[1], 0, csm, us_mask) - Kv
        M = X[0] + X[1] - A.mtimes(axb, 1, csm, us_mask) * t
        kesp = kes
        thetap = theta
        Xhp = Xh
    return M, X[0], X[1]


if __name__ == '__main__':
    import time

    Np = 4
    Nt = 15
    Nv = 7
    FE = 40
    PE = 40
    SPE = 40
    Nc = 10
    csm = np.ones((Nc, FE, PE, SPE))  # Nc FE PE SPE
    K = np.ones((Nv, Np, Nt, Nc, FE, PE, SPE))  # Nv Np Nt Nc FE PE SPE
    A = Eop()
    regL = 0.03
    regS = 0.06
    blk = 16
    it = 20

    for mode in ["LLR_POGM"]:
        print("START RECON.... METHOD:" + mode)
        st = time.time()
        csm = torch.as_tensor(np.ascontiguousarray(csm)).to(torch.complex64).to('cuda')
        for v in range(Nv):
            print("NOW VEncoding: " + str(v + 1) + "/" + str(Nv))
            Kv = torch.as_tensor(np.ascontiguousarray(K[v])).to(torch.complex64).to('cuda')
            us_mask = (torch.abs(Kv[:, :, 0:1, 0:1]) > 0).to(torch.float32).to('cuda')
            rcomb = torch.sum(k2i_torch(Kv, ax=[-3, -2, -1]) * torch.conj(csm), 2)
            regFactor = torch.max(torch.abs(rcomb))
            Kv /= regFactor

            if mode == 'GLR_ISTA':
                X = GLR_ISTA(A, Kv, csm, us_mask, regL, it)
            elif mode == 'GLR_FISTA':
                X = GLR_FISTA(A, Kv, csm, us_mask, regL, it)
            elif mode == 'GLR_POGM':
                X = GLR_POGM(A, Kv, csm, us_mask, regL, it)
            elif mode == "L+S":
                X, Lv, Sv = LplusS(A, Kv, csm, us_mask, regL, regS, it)
            elif mode == 'L+S_POGM':
                X, Lv, Sv = LplusS_POGM(A, Kv, csm, us_mask, regL, regS, it)
            elif mode == 'LLR_POGM':
                X = LLR_POGM(A, Kv, csm, us_mask, regL, it, blk)
        print("TIME(s):", time.time() - st)
