import argparse
import numpy as np
import pandas as pd
from blueprint.data_generator import create_ad_matrix
from blueprint.data_generator import create_rr_matrix
from blueprint.data_generator import set_obj
np.random.seed(0)


def set_args():
    # parse network data
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_num', type=int, default=50)
    parser.add_argument('--iata_file', type=str, default='../data/dataset/iata.csv')
    parser.add_argument('--passenger_file', type=str,
                        default='../data/dataset/passenger.csv')
    parser.add_argument('--route_file', type=str, default='../data/dataset/route.csv')
    parser.add_argument('--beta_max', type=float, default=0.05)
    parser.add_argument('--delta_max', type=float, default=0.1)
    parser.add_argument('--delta_min', type=float, default=0.08)
    return parser.parse_args()


if __name__ == '__main__':
    # set params and load data
    args = set_args()
    df_iata = pd.read_csv(args.iata_file)
    df_passenger = pd.read_csv(args.passenger_file).head(args.node_num)
    df_route = pd.read_csv(args.route_file)

    # create adjecent matrix
    df_ad_matrix = create_ad_matrix(args, df_iata, df_passenger, df_route)
    df_rr_matrix = create_rr_matrix(args)
    M, W, d, barx = set_obj(args, df_ad_matrix)

    # save data
    df_ad_matrix.to_csv('../data/dataset/ad_matrix.csv')
    np.save('../data/matrix/ad_matrix.npy', df_ad_matrix.values)
    np.save('../data/matrix/rr_matrix.npy', df_rr_matrix)
    np.save('../data/matrix/M.npy', M)
    np.save('../data/matrix/W.npy', W)
    np.save('../data/matrix/d.npy', d)
    np.save('../data/matrix/barx.npy', barx)
