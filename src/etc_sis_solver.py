import argparse
import numpy as np
import cvxpy as cp
from model.etc_sis_model import ETC_SIS
from utils.utils import achieving_objective
np.random.seed(0)
INF = 1e9
epsilon = 1e-15


def ly_param_solver(args, B, D):
    p_v = cp.Variable(args.node_num, pos=True)
    rc_v = cp.Variable(args.node_num)

    ly_cons1 = [rc_v == p_v @ (B.T - D)]
    ly_cons2 = [args.pubar <= p_v[i] for i in range(args.node_num)]
    ly_cons3 = [p_v[i] <= args.pbar for i in range(args.node_num)]
    ly_constraints = ly_cons1 + ly_cons2 + ly_cons3

    f_ly = 0
    for i in range(args.node_num):
        f_ly += rc_v[i]

    prob_lyapunov = cp.Problem(cp.Minimize(f_ly), ly_constraints)
    prob_lyapunov.solve(solver=cp.MOSEK)

    p = np.array(p_v.value)
    p = p / np.linalg.norm(p)
    return p


def set_args():
    # parse params
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=50)
    parser.add_argument('--ad_matrix_file', type=str, default='../data/matrix/ad_matrix.npy')
    parser.add_argument('--rr_matrix_file', type=str, default='../data/matrix/rr_matrix.npy')
    parser.add_argument('--M_matrix_file', type=str, default='../data/matrix/M.npy')
    parser.add_argument('--W_matrix_file', type=str, default='../data/matrix/W.npy')
    parser.add_argument('--d_matrix_file', type=str, default='../data/matrix/d.npy')
    parser.add_argument('--barx_matrix_file', type=str, default='../data/matrix/barx.npy')
    parser.add_argument('--pubar', type=float, default=0.0000001)
    parser.add_argument('--pbar', type=float, default=0.000000101)
    parser.add_argument('--kbar', type=float, default=0.7)
    return parser.parse_args()

if __name__ == '__main__':
    # set params and load data
    args = set_args()
    B = np.load(args.ad_matrix_file)
    D = np.load(args.rr_matrix_file)
    M = int(np.load(args.M_matrix_file))
    W = np.load(args.W_matrix_file)
    d = np.load(args.d_matrix_file)
    barx = np.load(args.barx_matrix_file)

    # design lyapunov params
    p = ly_param_solver(args, B, D)

    # analyse thetastar in the case of no input
    etc_sis = ETC_SIS(args=args, p=p, B=B, D=D, M=M, W=W, d=d)

    # analyze theta when no control input
    On = np.zeros((args.node_num, args.node_num))
    thetastar_noinput = etc_sis.analyze_theta(K=On, L=On, G=On, H=On)

    if achieving_objective(args, thetastar=thetastar_noinput, d=d, p=p, M=M, W=W) is False:
        print('Control objectives cannot be achieved without control inputs.')

        # obtain control gain
        Lstar, Kstar = etc_sis.control_gain_solver_gp()

        # obtain event-triggering gain
        sigmastar, etastar = etc_sis.triggered_parameter_solver_gp(Lstar, Kstar)

        # save data
        np.save('../data/matrix/L.npy', Lstar)
        np.save('../data/matrix/K.npy', Kstar)
        np.save('../data/matrix/G.npy', np.diag(sigmastar))
        np.save('../data/matrix/H.npy', np.diag(etastar))
