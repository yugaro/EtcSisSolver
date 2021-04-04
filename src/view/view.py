import math
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter
np.random.seed(10)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

class FixedOrderFormatter(ScalarFormatter):
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(
            self, useOffset=useOffset, useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        self.orderOfMagnitude = self._order_of_mag


def event_trigger_func(x, xk, sigma, eta):
    # define triggering rule
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data_all(args, B, D, L, K, G, H, W, barx, choice):
    # define propotion of infected pepole
    x = np.zeros([args.Time, args.node_num])
    x0 = np.random.rand(args.node_num)
    x[0] = x0
    xk = x0
    In = np.identity(args.node_num)

    # define event and objective list
    event = np.zeros([args.Time, args.node_num])
    d_table_list = np.array([barx for i in range(args.Time)])
    u_transition = np.zeros([args.Time - 1, args.node_num])
    v_transition = np.zeros([args.Time - 1, args.node_num])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(args.Time - 1):

        # # choice 1 has no control input
        if choice == 1:
            x[k + 1] = x[k] + args.h * \
                (-D.dot(x[k]) + (In - np.diag(x[k])).dot(B.T).dot(x[k]))

        # # In the case of using feedback controller
        else:
            for i in range(args.node_num):
                # # continuous controller
                if choice == 2 and event_trigger_func(x[k][i], xk[i], 0, 0) == 1:
                    xk[i] = x[k][i]
                    event[k + 1][i] = 1

                # # event-triggered controller
                elif choice == 3 and event_trigger_func(x[k][i], xk[i], G[i][i], H[i][i]) == 1:
                    xk[i] = x[k][i]
                    event[k + 1][i] = 1
            x[k + 1] = x[k] + args.h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
                In - np.diag(x[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x[k]))

    # plot data
    fig = plt.figure(figsize=(16, 9.7))
    ax1 = fig.add_axes((0, 0, 1, 1))
    cm = plt.cm.get_cmap('jet', args.node_num)
    for i in (range(args.node_num)):
        ax1.plot(x.T[i], lw=2, color=cm(i))
    ax1.set_xlabel(r'$t$', fontsize=60)
    ax1.set_ylabel(
        r'$x_i(t)$', fontsize=60)
    ax1.set_xticks([0, 200000, 400000, 600000, 800000])
    ax1.xaxis.set_major_formatter(FixedOrderFormatter(4, useMathText=True))
    ax1.xaxis.offsetText.set_fontsize(0)
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(4, 4))
    ax1.tick_params(axis='x', labelsize=60)
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels([r'$0$', r'$0.5$', r'$1$'])
    ax1.tick_params(axis='y', labelsize=60)
    ax1.grid(which='major', alpha=0.8, linestyle='dashed')

    if choice == 3:
        ax2 = fig.add_axes((0.3, 0.3, 0.65, 0.65))
        for i in (range(args.node_num)):
            ax2.plot(np.arange(400000, args.Time),
                     x.T[i][400000:], lw=2, color=cm(i))

        ax2.set_xticks([400000, 600000, 800000])
        ax2.xaxis.set_major_formatter(FixedOrderFormatter(4, useMathText=True))
        ax2.xaxis.offsetText.set_fontsize(0)
        ax2.ticklabel_format(style="sci", axis="x", scilimits=(4, 4))
        ax2.tick_params(axis='x', labelsize=60)
        # ax2.set_yscale('log')
        ax2.set_yticks([0, 0.05, 0.1])
        ax2.set_yticklabels([r'$0$', r'$0.05$', r'$0.1$'])
        ax2.tick_params(axis='y', labelsize=60)
        ax2.grid(which='major', alpha=0.6, linestyle='dotted')

    if choice == 1:
        fig.savefig("../image/x_traj_zero_all.pdf",
                    bbox_inches="tight", dpi=300)
    elif choice == 2:
        fig.savefig("../image/x_traj_con_all.pdf",
                    bbox_inches="tight", dpi=300)
    elif choice == 3:
        fig.savefig("../image/x_traj_etc_all.pdf",
                    bbox_inches="tight", dpi=300)


def plot_data_group(args, B, D, L, K, G, H, W, barx):

    # define propotion of infected pepole
    x_noinput = np.zeros([args.Time, args.node_num])
    x_continuous = np.zeros([args.Time, args.node_num])
    x_control = np.zeros([args.Time, args.node_num])
    x0 = np.random.rand(args.node_num)
    x_noinput[0] = x0
    x_continuous[0] = x0
    x_control[0] = x0
    xk = x0
    In = np.identity(args.node_num)

    # define event and objective list
    event = np.zeros([args.Time, args.node_num])
    d_table_list = np.array([barx for i in range(args.Time)])
    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(args.Time - 1):
        x_noinput[k + 1] = x_noinput[k] + args.h * \
            (-D.dot(x_noinput[k]) +
             (In - np.diag(x_noinput[k])).dot(B.T).dot(x_noinput[k]))
        x_continuous[k + 1] = x_continuous[k] + args.h * (-(D + K.dot(np.diag(x_continuous[k]))).dot(x_continuous[k]) + (
            In - np.diag(x_continuous[k])).dot(B.T - L.dot(np.diag(x_continuous[k])).T).dot(x_continuous[k]))
        for i in range(args.node_num):
            # # choice 3 is the case of event-triggered controller
            if event_trigger_func(x_control[k][i], xk[i], G[i][i], H[i][i]) == 1:
                xk[i] = x_control[k][i]
                event[k + 1][i] = 1
        x_control[k + 1] = x_control[k] + args.h * (-(D + K.dot(np.diag(xk))).dot(x_control[k]) + (
            In - np.diag(x_control[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x_control[k]))

    # compute the average of trajectories in each group
    for m in range(W.shape[0]):
        # subplot 1 is the transition data of x
        x_com_ave_noinput = 0
        x_com_ave_continuous = 0
        x_com_ave_control = 0
        community_member_num = 0
        for i in range(args.node_num):
            if W[m][i] == 1:
                x_com_ave_noinput += x_noinput.T[i]
                x_com_ave_continuous += x_continuous.T[i]
                x_com_ave_control += x_control.T[i]
                community_member_num += 1

        fig = plt.figure(figsize=(16, 9.7))
        ax = fig.add_axes((0, 0, 1, 1))

        ax.plot(x_com_ave_noinput / community_member_num, linestyle="dotted",
                lw=7, color='lime', label=r'Zero Control Input $(\mathcal{V}_{%d})$' % (m + 1), zorder=2)

        ax.plot(x_com_ave_continuous / community_member_num, linestyle="dashed",
                lw=7, color='dodgerblue', label=r'Continuous-Time Control $(\mathcal{V}_{%d})$' % (m + 1), zorder=3)

        ax.plot(x_com_ave_control / community_member_num, linestyle="solid",
                lw=7, color='crimson', label=r'Event-Triggered Control $(\mathcal{V}_{%d})$' % (m + 1), zorder=4)

        ax.plot(d_table_list.T[m], lw=7, linestyle="dashdot",
                label=r'Threshold $(\bar{x}_%d = %.2f)$' % (m, barx[m]), color='darkorange', zorder=1)

        # # # plot setting
        ax.set_xlabel(r'$t$', fontsize=60)
        ax.set_ylabel(
            r'$\frac{1}{|\mathcal{V}_%d|}\sum_{i\in \mathcal{V}_%d} x_i(t)$' % (m + 1, m + 1), fontsize=60)
        ax.set_xticks([0, 200000, 400000, 600000, 800000])
        ax.xaxis.set_major_formatter(FixedOrderFormatter(4, useMathText=True))
        ax.xaxis.offsetText.set_fontsize(0)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(4, 4))
        ax.tick_params(axis='x', labelsize=60)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_yticklabels([r'$0$', r'$0.25$', r'$0.5$'])
        ax.set_yticks([0.1], minor=True)
        ax.set_yticklabels([r'$0.1$'], minor=True)
        ax.set_ylim(0, 0.58)
        ax.tick_params(axis='y', labelsize=60, which='both')
        ax.legend(loc="upper right", bbox_to_anchor=(
            1.0, 1.0), borderaxespad=0, fontsize=48, ncol=1)
        ax.grid(which='major', alpha=0.8, linestyle='dashed')
        fig.savefig("../image/x_traj_group{}.pdf".format(m + 1),
                    bbox_inches="tight", dpi=300)
        plt.close()


def plot_data_gain(args, B, D, L, K, G, H, W, barx, group_part=1):
    x_event = np.zeros([args.Time, args.node_num])
    x0 = np.random.rand(args.node_num)
    x_event[0] = x0
    xk = x0
    In = np.identity(args.node_num)
    args.Time = int(args.Time / 2)

    # define event and objective list
    event = np.zeros([args.Time, args.node_num])

    # create control input series (feedback, event-trigger)
    u_transition_event = np.zeros([args.Time - 1, args.node_num])
    v_transition_event = np.zeros(
        [args.Time - 1, args.node_num, args.node_num])

    # define inter-time events
    inter_time_events = np.zeros([args.Time, args.node_num])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(args.Time - 1):
        for i in range(args.node_num):
            # # event-triggered control
            if event_trigger_func(x_event[k][i], xk[i], G[i][i], H[i][i]) == 1:
                xk[i] = x_event[k][i]
                event[k + 1][i] = 1
            v_transition_event[k][i] = L[i] * xk
        x_event[k + 1] = x_event[k] + args.h * (-(D + K.dot(np.diag(xk))).dot(x_event[k]) + (
            In - np.diag(x_event[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x_event[k]))
        # # # record event-triggered control input
        u_transition_event[k] = K.dot(xk)

    for i in range(args.node_num):
        triggering_tmp = 0
        for k in range(args.Time):
            if event[k][i] == 1:
                inter_time_events[k][i] = k - triggering_tmp
                triggering_tmp = k
            else:
                inter_time_events[k][i] = None

    # subplot U (event)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    cm = plt.cm.get_cmap('jet', args.node_num)
    for i in range(args.node_num):
        if W[group_part - 1][i] == 1:
            ax.plot(u_transition_event.T[i],
                    lw=4, color=cm(i), alpha=1)
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$u_i(t)$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.ylim(0, u_transition_event.max())
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.tight_layout()
    plt.grid(which='major', alpha=0.8, linestyle='dashed')
    plt.savefig(
        "../image/u_traj_group{}.pdf".format(group_part), dpi=300)

    # subplot V (event)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    v_argmax = np.argmax(v_transition_event[0], axis=0)
    for i in range(args.node_num):
        if W[group_part - 1][i] == 1:
            ax.plot(v_transition_event.T[i][v_argmax[i]],
                    lw=4, color=cm(i), alpha=1)
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$v_{ij}(t)$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.ylim(0, v_transition_event.max())
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.tight_layout()
    plt.grid(which='major', alpha=0.8, linestyle='dashed')
    plt.savefig(
        "../image/v_traj_group{}.pdf".format(group_part), dpi=300)

    # subplot inter-event time
    fig, ax = plt.subplots(figsize=(16, 9.7))
    for i in (range(args.node_num)):
        if W[group_part - 1][i] == 1:
            ax.scatter(np.arange(args.Time), inter_time_events.T[i], color=cm(i), s=100, alpha=1)
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$t_{\ell + 1}^i - t_\ell^i$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=60)
    plt.tight_layout()
    plt.grid()
    plt.savefig("../image/iet_traj_group{}.pdf".format(group_part), dpi=300)


def plot_data_compare(args, B, D, L, K, G, H, W, barx, group_part=1):
    x_event = np.zeros([args.Time, args.node_num])
    x_continuous = np.zeros([args.Time, args.node_num])
    x0 = np.random.rand(args.node_num)
    x_event[0] = x0
    x_continuous[0] = x0
    xk = x0
    In = np.identity(args.node_num)

    # define event and objective list
    event = np.zeros([args.Time, args.node_num])

    # create control input series (feedback, event-trigger)
    u_transition_event = np.zeros([args.Time - 1, args.node_num])
    u_transition_continuous = np.zeros([args.Time - 1, args.node_num])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(args.Time - 1):
        x_continuous[k + 1] = x_continuous[k] + args.h * (-(D + K.dot(np.diag(x_continuous[k]))).dot(x_continuous[k]) + (
            In - np.diag(x_continuous[k])).dot(B.T - L.dot(np.diag(x_continuous[k])).T).dot(x_continuous[k]))
        for i in range(args.node_num):
            # # event-triggered control
            if event_trigger_func(x_event[k][i], xk[i], G[i][i], H[i][i]) == 1:
                xk[i] = x_event[k][i]
                event[k + 1][i] = 1
        x_event[k + 1] = x_event[k] + args.h * (-(D + K.dot(np.diag(xk))).dot(x_event[k]) + (
            In - np.diag(x_event[k])).dot(B.T - L.dot(np.diag(xk)).T).dot(x_event[k]))
        # # record event-triggered control input
        u_transition_event[k] = K.dot(xk)
        u_transition_continuous[k] = K.dot(x_continuous[k])

    # subplot U (event)
    fig, ax = plt.subplots(figsize=(16, 9.7))
    cm = plt.cm.get_cmap('jet', args.node_num)
    for i in range(args.node_num):
        if W[group_part - 1][i] == 1:
            ax.plot(u_transition_event.T[i],
                    lw=4, color='magenta', alpha=1, label='event-triggered')
            ax.plot(u_transition_continuous.T[i],
                    lw=4, color='cyan', alpha=1, label='continuous')
    plt.xlabel(r'$t$', fontsize=60)
    plt.ylabel(
        r'$u_i(t)$', fontsize=60)
    plt.xticks([0, 100000, 200000, 300000, 400000])
    ax.xaxis.offsetText.set_fontsize(0)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))
    plt.setp(ax.get_xticklabels(), fontsize=60)
    plt.ylim(0, u_transition_event.max())
    ax.yaxis.offsetText.set_fontsize(60)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.setp(ax.get_yticklabels(), fontsize=60)
    plt.tight_layout()
    plt.grid(which='major', alpha=0.8, linestyle='dashed')
    plt.savefig(
        "../image/u_traj_event_continuous_group{}.pdf".format(group_part), dpi=300)
