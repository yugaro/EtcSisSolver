import numpy as np
import cvxpy as cp
from controller.control_gain import control_gain_constant
from controller.control_gain import control_gain_constraint
from controller.triggering_gain import triggering_gain_constant
from controller.triggering_gain import triggering_gain_constraint
np.random.seed(0)
INF = 1e9
epsilon = 1e-15


class ETC_SIS:
    def __init__(self, args, p, B, D, M, W, d):
        self.args = args
        self.n = args.node_num
        self.On = np.zeros((args.node_num))
        self.In = np.identity(args.node_num)
        self.p = p
        self.B = B
        self.D = D
        self.M = M
        self.W = W
        self.d = d
        self.barL = B - epsilon
        self.barK = np.identity(args.node_num) * args.kbar

    def analyze_theta(self, K, L, G, H):
        # define variables of the state of nodes
        x = cp.Variable(self.n, pos=True)

        # define parameter of theorem 1
        s = (K + L.T).dot(self.In - G).dot(self.p)
        S = np.diag(s)
        Q = S + 1 / 2 * np.diag(self.p).dot(L.T).dot(self.In - G).dot(G + H)
        Q = (Q.T + Q) / 2
        r = (self.B.T - self.D + (K + L.T).dot(H)).dot(self.p)

        # define constraint in theorem 1
        if np.all(Q == 0):
            constranit_theta = [- r.T @ x <= 0,
                                0 <= x, x <= 1]
        else:
            constranit_theta = [cp.quad_form(x, Q) - r.T @ x <= 0,
                                0 <= x, x <= 1]

        # define objective function in theorem 1
        theta = self.p.T @ x

        # solve program of theorem 1 and caluculate theta*
        prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
        prob_theta.solve(solver=cp.MOSEK)

        return prob_theta.value

    def control_gain_solver_gp(self):
        # define varialbe
        tildeL = cp.Variable((self.n, self.n), pos=True)
        tildeK = cp.Variable((self.n, self.n), pos=True)
        s = cp.Variable((self.M, self.n), pos=True)
        xi = cp.Variable((self.M, 1), pos=True)

        # define constant
        rc, c1, c2 = control_gain_constant(
            n=self.n, p=self.p, B=self.B, D=self.D, M=self.M, W=self.W, d=self.d, barL=self.barL, barK=self.barK)

        # create constraints
        gp_consts_c = control_gain_constraint(n=self.n, p=self.p, B=self.B, M=self.M, W=self.W, d=self.d,
                                              barL=self.barL, barK=self.barK, rc=rc, c1=c1, c2=c2, tildeL=tildeL, tildeK=tildeK, s=s, xi=xi)

        # create objective func and solve GP (control parameters)
        gp_fc = 1
        for i in range(self.n):
            gp_fc += self.barK[i][i] * 10000000000000 / tildeK[i][i]
            for j in range(self.n):
                if self.B[i][j] != 0:
                    gp_fc += self.barL[i][j] / tildeL[i][j]

        gp_prob_c = cp.Problem(cp.Maximize(1 / gp_fc), gp_consts_c)
        gp_prob_c.solve(gp=True, solver=cp.MOSEK)
        print("GP status (control parameters):", gp_prob_c.status)

        # get value of K and L
        Lstar = self.barL - np.array(tildeL.value)
        Kstar = self.barK - np.array(tildeK.value)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    Kstar[i][j] = 0
                if self.B[i][j] == 0:
                    Lstar[i][j] = 0
        return Lstar, Kstar

    def triggered_parameter_solver_gp(self, Lstar, Kstar):
        # define variable
        sigma = cp.Variable(self.n, pos=True)
        eta = cp.Variable(self.n, pos=True)
        r = cp.Variable((self.M, self.n), pos=True)
        s = cp.Variable((self.M, self.n), pos=True)
        xi1 = cp.Variable((self.M, 1), pos=True)
        xi2 = cp.Variable((self.M, self.n), pos=True)

        # define constant
        rc, c3 = triggering_gain_constant(
            n=self.n, p=self.p, B=self.B, D=self.D, M=self.M, W=self.W, Lstar=Lstar, Kstar=Kstar)

        # create constraints
        gp_consts_t = triggering_gain_constraint(
            n=self.n, p=self.p, B=self.B, M=self.M, W=self.W, d=self.d, rc=rc, c3=c3, sigma=sigma, eta=eta, r=r, s=s, xi1=xi1, xi2=xi2)

        # create objective funcition and solve GP (triggered paramters)
        gp_ft = 1
        for i in range(self.n):
            gp_ft *= (sigma[i]) * (eta[i])
        gp_prob_e = cp.Problem(cp.Maximize(gp_ft), gp_consts_t)
        gp_prob_e.solve(gp=True, solver=cp.MOSEK)
        print("GP status (event-triggered paramters) :", gp_prob_e.status)

        # get value of sigma and eta
        sigmastar = np.array(sigma.value)
        etastar = np.array(eta.value)
        return sigmastar, etastar
