import math

import numpy as np


class DeEPCAAlgorithm:
    def __init__(self, data, iterations, K, num_nodes, initial_est, ground_truth):
        self.data = data  # Data samples
        self.num_itr = iterations  # Number of iterations
        self.K = K  # Dimension of eigenspace to be estimated
        self.n = num_nodes  # Number of nodes
        self.X_init = initial_est  # Initial estimate of K-dimensional eigenspace (dxK)
        self.X_gt = ground_truth  # True K-dimensional eigenspace (dxK)

    def DeEPCA(self, WW, K_fastmix):
        N = self.data.shape[1]
        S = np.tile(self.X_init.T, (self.n, 1))
        X = np.tile(self.X_init.T, (self.n, 1))
        X_prev = np.tile(self.X_init.T, (self.n, 1))
        angle_deepca = self.dist_subspace(self.X_gt, self.X_init)
        angle_deepca1 = np.tile(angle_deepca, (K_fastmix, 1))

        Cy_cell = np.zeros((self.n,), dtype=object)
        s = math.floor(N / self.n)
        for i in range(self.n):  # Loop over nodes
            Yi = self.data[:, i * s:(i + 1) * s]
            Cy_cell[i] = (1 / s) * np.dot(Yi, Yi.T)

        # Initial mixing and projection
        for i in range(self.n):
            S1 = S[i * self.K:(i + 1) * self.K, :].T + np.dot(Cy_cell[i], self.X_init) - self.X_init
            S[i * self.K:(i + 1) * self.K, :] = S1.T
        S = self.FastMix(S, K_fastmix, WW, self.K)

        # Update and calculate errors
        Tp = int(self.num_itr / K_fastmix)
        for itr in range(Tp):
            for i in range(self.n):
                Xx = X[i * self.K:(i + 1) * self.K, :].T
                Xx_prev = X_prev[i * self.K:(i + 1) * self.K, :].T
                S1 = S[i * self.K:(i + 1) * self.K, :].T + np.dot(Cy_cell[i], Xx) - np.dot(Cy_cell[i], Xx_prev)
                S[i * self.K:(i + 1) * self.K, :] = S1.T
            S = self.FastMix(S, K_fastmix, WW, self.K)
            err = 0
            for i in range(self.n):
                S2 = S[i * self.K:(i + 1) * self.K, :].T
                X1, _ = np.linalg.qr(S2)
                X1 = self.SignAdjust(X1, self.X_init)
                X_prev[i * self.K:(i + 1) * self.K, :] = X[i * self.K:(i + 1) * self.K, :]
                X[i * self.K:(i + 1) * self.K, :] = X1.T
                err += self.dist_subspace(self.X_gt, X1)
            angle_deepca1 = np.append(angle_deepca1, np.tile(err / self.n, (K_fastmix, 1)))
        return angle_deepca1

    def FastMix(self, S, K, WW, dim):
        S_prev = S
        eig_w = np.linalg.eigvalsh(WW)
        eig_w1 = np.unique(eig_w)
        eta_w = (1 - np.sqrt(1 - eig_w1[-dim - 1] ** 2)) / (1 + np.sqrt(1 - eig_w1[-dim - 1] ** 2))
        for _ in range(K):
            S1 = (1 + eta_w) * np.dot(WW, S) - eta_w * S_prev
            S_prev = S
            S = S1
        return S

    def SignAdjust(self, X, X0):
        for i in range(X0.shape[1]):
            if np.dot(X[:, i].T, X0[:, i]) < 0:
                X[:, i] = -X[:, i]
        return X

    def dist_subspace(self, X, Y):
        X = X / np.linalg.norm(X, axis=0)
        Y = Y / np.linalg.norm(Y, axis=0)
        M = np.matmul(X.T, Y)
        sine_angle = 1 - np.diag(M) ** 2
        return np.sum(sine_angle) / X.shape[1]
