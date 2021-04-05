import numpy as np
from utils.utils import p_min_supp_w
np.random.seed(0)
INF = 1e9
epsilon = 1e-15


def triggering_gain_constant(n, p, B, D, M, W, Lstar, Kstar):
    # define constant
    # # rc = p.T (B.T - D)
    rc = p.T.dot(B.T - D)

    # # c3 = p K* + sum(p L*)
    c3 = np.zeros((M, n))
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                c3[m][i] = p[i] * Kstar[i][i]
                for j in range(n):
                    if B[i][j] != 0:
                        c3[m][i] += p[j] * Lstar[i][j]
    return rc, c3


def triggering_gain_constraint(n, p, B, M, W, d, rc, c3, sigma, eta, r, s, xi1, xi2):
    # create constraints
    # # sigma + epsilon <= 1
    gp_const1_t = [sigma[i] + epsilon <= 1 for i in range(n)]

    # # eta + epsilon <= 1
    gp_const2_t = [eta[i] + epsilon <= 1 for i in range(n)]

    # # s / tildesigma <= c3
    gp_const3_t = [sigma[i] + 1 / xi2[m][i] <=
                   1 for m in range(M) for i in range(n) if W[m][i] == 1]
    gp_const3_t += [s[m][i] * xi2[m][i] <= c3[m][i]
                    for m in range(M) for i in range(n) if W[m][i] == 1]

    # # max{0, rc} + c3 * eta <= r
    gp_const4_t = []
    for m in range(M):
        for i in range(n):
            if rc[i] >= 0 and W[m][i] == 1:
                gp_const4_t += [rc[i] + c3[m][i] * eta[i] <= r[m][i]]
            elif rc[i] < 0 and -rc[i] < c3[m][i] and W[m][i] == 1:
                gp_const4_t += [eta[i] == -rc[i] / c3[m][i]]
            elif W[m][i] == 1:
                gp_const4_t += [c3[m][i] * eta[i] <= r[m][i]]

    # # (sum p^2 / s) * (sum r^2 / s) <= xi1
    gp_const5_t = []
    for m in range(M):
        tmp1_const5_t = 0
        tmp2_const5_t = 0
        for i in range(n):
            if W[m][i] == 1:
                tmp1_const5_t += p[i] ** 2 / s[m][i]
                tmp2_const5_t += r[m][i] ** 2 / s[m][i]
        gp_const5_t += [tmp1_const5_t * tmp2_const5_t <= xi1[m]]

    # # xi ** 0.5 + sum p * r / s <= 2 * p_m * d
    gp_const6_t = []
    p_m = p_min_supp_w(n, p, M, W)
    for m in range(M):
        tmp_const6_t = 0
        for i in range(n):
            if W[m][i] == 1:
                tmp_const6_t += p[i] * r[m][i] / s[m][i]
        gp_const6_t += [xi1[m]**0.5 + tmp_const6_t <= 2 * p_m[m] * d[m]]

    # # configure gp constraints
    gp_consts_t = gp_const1_t + gp_const2_t + gp_const3_t + \
        gp_const4_t + gp_const5_t + gp_const6_t

    return gp_consts_t
