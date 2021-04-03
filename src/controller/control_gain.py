import numpy as np
from utils.utils import p_min_supp_w
np.random.seed(0)
INF = 1e9
epsilon = 1e-15


def control_gain_constant(n, p, B, D, M, W, d, barL, barK):
    # define constant
    # # rc = p.T (B.T - D)
    rc = (B.T - D).dot(p)
    # # c1 = p barK + sum(p barL)
    c1 = np.zeros((M, n))
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                c1[m][i] = p[i] * barK[i][i]
                for j in range(n):
                    if B[i][j] != 0:
                        c1[m][i] += p[j] * barL[i][j]
    # # c2 = 2 p_m *d - sum_{rc < 0} p * rc / c1
    c2 = np.zeros(M)
    p_m = p_min_supp_w(n, p, M, W)
    for m in range(M):
        c2[m] = 2 * p[m] * d[m]
        for i in range(n):
            if rc[i] < 0 and W[m][i] == 1:
                c2[m] -= p[i] * rc[i] / c1[m][i]
    return rc, c1, c2


def control_gain_constraint(n, p, B, M, W, d, barL, barK, rc, c1, c2, tildeL, tildeK, s, xi):
    # create constraints
    # # tildeK + epsilon <= barK
    gp_const1_c = [tildeK[i][i] + epsilon <= barK[i][i]
                   for i in range(n)]

    # # tildeL + epsilon <= barL
    gp_const2_c = [tildeL[i][j] + epsilon <= barL[i][j]
                   for i in range(n) for j in range(n) if B[i][j] != 0]

    # # s + p tildeK + sum(p tildeL) <= c1
    gp_const3_c = []
    for m in range(M):
        for i in range(n):
            if W[m][i] == 1:
                tmp_const3_c = 0
                for j in range(n):
                    if B[i][j] != 0:
                        tmp_const3_c += p[j] * tildeL[i][j]
                gp_const3_c += [s[m][i] + p[i] * tildeK[i]
                                [i] + tmp_const3_c <= c1[m][i]]

    # # sum(p^2/s) * sum(r^2/s) <= xi
    gp_const4_c = []
    for m in range(M):
        tmp1_const4_c = 0
        tmp2_const4_c = 0
        for i in range(n):
            if W[m][i] == 1:
                tmp1_const4_c += p[i]**2 / s[m][i]
                tmp2_const4_c += rc[i]**2 / s[m][i]
        gp_const4_c += [tmp1_const4_c * tmp2_const4_c <= xi[m]]

    # # xi + sum_{rc >= 0}( p * rc / s) <= c2
    gp_const5_c = []
    for m in range(M):
        tmp_const5_c = 0
        for i in range(n):
            if rc[i] >= 0 and W[m][i] == 1:
                tmp_const5_c += p[i] * rc[i] / s[m][i]
        gp_const5_c += [xi[m]**0.5 + tmp_const5_c <= c2[m]]

    # configure gp constraints of control paramters
    gp_consts_c = gp_const1_c + gp_const2_c + \
        gp_const3_c + gp_const4_c + gp_const5_c

    return gp_consts_c
