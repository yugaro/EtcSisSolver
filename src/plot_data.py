import argparse
import numpy as np
from view.view import plot_data_all
from view.view import plot_data_group
from view.view import plot_data_gain
np.random.seed(11)

def set_args():
    # parse params
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=50)
    parser.add_argument('--Time', type=float, default=800000)
    parser.add_argument('--h', type=float, default=0.0001)
    parser.add_argument('--ad_matrix_file', type=str,
                        default='../data/matrix/ad_matrix.npy')
    parser.add_argument('--rr_matrix_file', type=str,
                        default='../data/matrix/rr_matrix.npy')
    parser.add_argument('--K_matrix_file', type=str,
                        default='../data/matrix/K.npy')
    parser.add_argument('--L_matrix_file', type=str,
                        default='../data/matrix/L.npy')
    parser.add_argument('--G_matrix_file', type=str,
                        default='../data/matrix/G.npy')
    parser.add_argument('--H_matrix_file', type=str,
                        default='../data/matrix/H.npy')
    parser.add_argument('--M_matrix_file', type=str,
                        default='../data/matrix/M.npy')
    parser.add_argument('--W_matrix_file', type=str,
                        default='../data/matrix/W.npy')
    parser.add_argument('--d_matrix_file', type=str,
                        default='../data/matrix/d.npy')
    parser.add_argument('--barx_matrix_file', type=str,
                        default='../data/matrix/barx.npy')

    return parser.parse_args()

if __name__ == '__main__':
    # set params and load data
    args = set_args()
    B = np.load(args.ad_matrix_file)
    D = np.load(args.rr_matrix_file)
    L = np.load(args.L_matrix_file)
    K = np.load(args.K_matrix_file)
    G = np.load(args.G_matrix_file)
    H = np.load(args.H_matrix_file)
    M = int(np.load(args.M_matrix_file))
    W = np.load(args.W_matrix_file)
    d = np.load(args.d_matrix_file)
    barx = np.load(args.barx_matrix_file)

    # plot data
    plot_data_all(args, B, D, L, K, G, H, W, barx, choice=3)
    plot_data_group(args, B, D, L, K, G, H, W, barx)
    plot_data_gain(args, B, D, L, K, G, H, W, barx, group_part=1)
