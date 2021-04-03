import numpy as np
INF = 1e9

def p_min_supp_w(n, p, M, W):
    # caluculate p_min in each target
    p_min = []
    for m in range(M):
        tmp_p = INF
        for j in range(n):
            if W[m][j] == 1 and tmp_p > p[j]:
                tmp_p = p[j]
        p_min += [tmp_p]
    return p_min


def achieving_objective(args, thetastar, d, p, M, W):
    p_min = p_min_supp_w(args.node_num, p, M, W)
    judge_noinput = [thetastar <= d[m] * p_min[m] for m in range(M)]
    if np.all(judge_noinput):
        return True
    else:
        return False
