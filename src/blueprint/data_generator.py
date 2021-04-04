import pandas as pd
import numpy as np
import community
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
np.random.seed(0)
colors = ["mediumblue", "brown", "forestgreen"]
cmap = ListedColormap(colors, name="custom")


def create_ad_matrix(args, df_iata, df_passenger, df_route):
    for index, row in df_passenger.iterrows():
        df_passenger.at[index, 'IATA'] = df_iata[df_iata['Airport Name']
                                                 == row['Airport Name']]['IATA'].values[0]

    df_source = pd.DataFrame()
    df_source_dest = pd.DataFrame()
    for index, row in df_passenger.iterrows():
        df_source = pd.concat(
            [df_source, df_route[df_route["Source airport"] == str(row["IATA"])]])
    for index, row in df_passenger.iterrows():
        df_source_dest = pd.concat(
            [df_source_dest, df_source[df_source["Destination airport"] == str(row["IATA"])]])

    df_ad_matrix = pd.DataFrame(
        index=df_passenger['IATA'].values, columns=df_passenger['IATA'].values)
    df_ad_matrix = df_ad_matrix.fillna(0)
    for index, row in df_source_dest.iterrows():
        df_ad_matrix.at[row['Source airport'], row['Destination airport']] += 1
    for index, row in df_ad_matrix.iterrows():
        df_ad_matrix[index] = df_ad_matrix[index].values * \
            df_passenger[df_passenger['IATA'] == index]['Passenger'].values[0]
    df_ad_matrix = (df_ad_matrix - np.nanmin(df_ad_matrix.values)) * args.beta_max / \
        (np.nanmax(df_ad_matrix.values - np.nanmin(df_ad_matrix.values)))
    for index, row in df_ad_matrix.iterrows():
        df_ad_matrix.at[index, index] = df_passenger[df_passenger['IATA'] == index]['Population'].values[0] * \
            args.beta_max * 0.1 / df_passenger['Population'].values.max()
    return df_ad_matrix


def create_rr_matrix(args):
    return np.diag(np.sort((args.delta_max - args.delta_min) *
                           np.random.rand(args.node_num) + args.delta_min)[::-1])


def set_obj(args, Ad_matrix):
    BB = Ad_matrix.T.values
    edge_list = [(i, j, BB[i][j]) for i in range(args.node_num)
                 for j in range(args.node_num) if BB[i][j] != 0]
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)

    # create community
    partition = community.best_partition(G, weight='weight', resolution=1.062214671, random_state=21)
    partition = dict(sorted(partition.items()))

    # choice target community
    group_list = [[2], [0], [1]]

    # count the number of control objectives
    M = len(group_list)

    # define threshold of each node in the target
    barx = np.array([0.08, 0.1, 0.06])

    # define target nodes according to community
    W = np.zeros((M, args.node_num))
    d = np.zeros(M)
    for m in range(M):
        for i in range(args.node_num):
            if partition[i] in group_list[m]:
                W[m][i] = 1
                d[m] += barx[m]

    # define node size
    eigv_cent = nx.eigenvector_centrality_numpy(G)
    node_size = np.array([(size ** 4) for size in list(eigv_cent.values())]) * 20000000

    # define node color
    node_color = [partition[i] for i in G.nodes()]

    # define edge width
    width = np.array([d['weight'] for (u, v, d) in G.edges(data=True)])
    width_std = 14 * (((width - min(width)) / (max(width) - min(width)))) + 0.5

    # define label name
    node_labels = {i: key for i, key in zip(
        np.array(range(args.node_num)), list(Ad_matrix.index))}

    # plot graph
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_axes((0, 0, 1, 1))
    pos = nx.nx_agraph.graphviz_layout(G, prog="sfdp")

    # set drawing nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_size,
        node_color=node_color,
        cmap=cmap
    )

    # set drawing labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=55
    )

    # set drawing edges
    nx.draw_networkx_edges(
        G,
        pos,
        width=width_std
    )

    plt.axes(ax).set_axis_off()
    fig.savefig('../image/air_transport_network.png', tight_layout=True)

    return M, W, d, barx
