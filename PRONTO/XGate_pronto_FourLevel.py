# -*- coding: utf-8 -*-
"""

Adapted from the user's original state-to-state PRONTO code.
Main change:
- optimize a quantum X gate in the qubit subspace {|1>,|2>}
- simultaneous trajectories:
    |1> -> |2>
    |2> -> |1>
- line-search candidates are projected with the PRONTO regulator:
    u(t) = mu(t) - K_r(t)[x(t)-alpha(t)]

Basis convention:
|1> -> index 0
|2> -> index 1
|3> -> index 2
|4> -> index 3
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit


# =========================
# Global parameters
# =========================

N_LEVELS = 4
nmaxE = N_LEVELS
nvar = 2 * nmaxE
NCTRL = 2  # Omega_p, Omega_s

Ntmax = 5000
tfinal = 3.0
deltat = tfinal / float(Ntmax)

tol = 1e-10
# stopping criterion on terminal gate error, same as the 3-level code
mfunc_tol = 1.0e-3
sig = 0.2             # leakage penalty
RMIN = 1e-12

maxls = 200
max_riccati_tries = 12
max_iter = 120


delta = 0.0
Delta3 = 0.0
Delta4 = 6.79

sigma3 = 1.0
sigma4 = 1.0



# PRONTO projection regulator parameters
# Kr(t) is computed from an auxiliary Riccati equation:
#   -Pr_dot = A^T Pr + Pr A - Kr^T Rr Kr + Qr
#   Kr      = Rr^{-1} B^T Pr
kr_q = 1.0
kr_r = 1.0
kr_p = 1.0

# Output configuration
#OUT_PREFIX = "Xgate_4level_PRONTO_phase"
#OUT_DIR = "figs_Xgate_4level_PRONTO_phase"

# Haar random-state fidelity configuration
N_HAAR = 500
HAAR_SEED = 1234


# =========================
# Basic linear algebra kernels
# =========================

@njit(fastmath=True)
def matvec(A, x, y):
    n = A.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        y[i] = s


@njit(fastmath=True)
def matvec_T(A, x, y):
    n = A.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[j, i] * x[j]
        y[i] = s


@njit(fastmath=True)
def matmul(A, B, C):
    n = A.shape[0]
    p = B.shape[1]
    kdim = A.shape[1]
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(kdim):
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit(fastmath=True)
def symmetrize(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            v = 0.5 * (A[i, j] + A[j, i])
            A[i, j] = v
            A[j, i] = v


@njit(fastmath=True)
def is_spd_cholesky(A):
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 1e-14:
                    return False
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return True


@njit(fastmath=True)
def project_to_spd(A, lambda0, max_tries):
    symmetrize(A)
    if is_spd_cholesky(A):
        return
    lam = lambda0
    n = A.shape[0]
    for _ in range(max_tries):
        for i in range(n):
            A[i, i] += lam
        if is_spd_cholesky(A):
            return
        lam *= 2.0


@njit(fastmath=True)
def int_simp(points, delta_t):
    Np = points.shape[0] - 1
    soma = 0.0
    Nq = Np // 2
    for k in range(1, Nq + 1):
        l = 2 * k - 2
        soma += points[l] + 4.0 * points[l + 1] + points[l + 2]
    return delta_t * soma / 3.0


# =========================
# Small helpers for 2 controls
# =========================

@njit(fastmath=True)
def inv2(A, Ainv):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-18:
        det = 1e-18
    invdet = 1.0 / det
    Ainv[0, 0] =  A[1, 1] * invdet
    Ainv[0, 1] = -A[0, 1] * invdet
    Ainv[1, 0] = -A[1, 0] * invdet
    Ainv[1, 1] =  A[0, 0] * invdet


@njit(fastmath=True)
def quadform2(u, R):
    return (
        u[0] * (R[0, 0] * u[0] + R[0, 1] * u[1]) +
        u[1] * (R[1, 0] * u[0] + R[1, 1] * u[1])
    )


@njit(fastmath=True)
def BK_matrix(B, K, out):
    # B: (n,2), K: (2,n), out: (n,n) = B @ K
    n = B.shape[0]
    for i in range(n):
        for j in range(n):
            out[i, j] = B[i, 0] * K[0, j] + B[i, 1] * K[1, j]


@njit(fastmath=True)
def BT_P(B, P, out):
    # out = B^T @ P ; B: (n,2), P: (n,n), out: (2,n)
    n = B.shape[0]
    for a in range(2):
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += B[i, a] * P[i, j]
            out[a, j] = s


@njit(fastmath=True)
def corr_term(V, Rinv, out):
    # out = V^T @ Rinv @ V ; V: (2,n)
    n = V.shape[1]
    RV = np.empty((2, n), dtype=np.float64)

    for a in range(2):
        for j in range(n):
            RV[a, j] = Rinv[a, 0] * V[0, j] + Rinv[a, 1] * V[1, j]

    for i in range(n):
        for j in range(n):
            out[i, j] = V[0, i] * RV[0, j] + V[1, i] * RV[1, j]


@njit(fastmath=True)
def KTr(K, r, out):
    # out = K^T @ r ; K: (2,n), r: (2,)
    n = K.shape[1]
    for i in range(n):
        out[i] = K[0, i] * r[0] + K[1, i] * r[1]


@njit(fastmath=True)
def Bnu(B, nu, out):
    # out = B @ nu ; B: (n,2), nu: (2,)
    n = B.shape[0]
    for i in range(n):
        out[i] = B[i, 0] * nu[0] + B[i, 1] * nu[1]


@njit(fastmath=True)
def Kz(K, z, out):
    # out = K @ z ; K: (2,n), z: (n,)
    n = K.shape[1]
    s0 = 0.0
    s1 = 0.0
    for i in range(n):
        s0 += K[0, i] * z[i]
        s1 += K[1, i] * z[i]
    out[0] = s0
    out[1] = s1


@njit(fastmath=True)
def maxabs_controls(U):
    m = 0.0
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            v = abs(U[i, j])
            if v > m:
                m = v
    return m


@njit(fastmath=True)
def state_norm_squared(x):
    s = 0.0
    for i in range(x.shape[0]):
        s += x[i] * x[i]
    return s


@njit(fastmath=True)
def max_norm_deviation(psi):
    dev = 0.0
    for nt in range(psi.shape[0]):
        v = abs(state_norm_squared(psi[nt]) - 1.0)
        if v > dev:
            dev = v
    return dev


@njit(fastmath=True)
def max_norm_deviation_multi(psi_all):
    dev = 0.0
    for s in range(psi_all.shape[0]):
        d = max_norm_deviation(psi_all[s])
        if d > dev:
            dev = d
    return dev


@njit(fastmath=True)
def max_norm_rows_general(A):
    m = 0.0
    for i in range(A.shape[0]):
        s = 0.0
        for j in range(A.shape[1]):
            s += A[i, j] * A[i, j]
        v = np.sqrt(s)
        if v > m:
            m = v
    return m


# =========================
# PRONTO projection regulator Kr(t) for stacked gate trajectories
# =========================

@njit(fastmath=True)
def stack_multi_state_traj(psi_all):
    """Stack psi_all[s, nt, :] into Xtraj[nt, s*n:(s+1)*n]."""
    ntraj = psi_all.shape[0]
    ntot = psi_all.shape[1]
    n = psi_all.shape[2]
    Xtraj = np.zeros((ntot, ntraj * n), dtype=np.float64)
    for s in range(ntraj):
        Xtraj[:, s*n:(s + 1)*n] = psi_all[s]
    return Xtraj


@njit(fastmath=True)
def unstack_multi_state_traj(Xtraj, ntraj, n):
    """Unstack Xtraj[nt, :] into psi_all[s, nt, :]."""
    psi_all = np.zeros((ntraj, Xtraj.shape[0], n), dtype=np.float64)
    for s in range(ntraj):
        psi_all[s] = Xtraj[:, s*n:(s + 1)*n]
    return psi_all


@njit(fastmath=True)
def stack_initial_states(psi0_all):
    ntraj = psi0_all.shape[0]
    n = psi0_all.shape[1]
    X0 = np.zeros(ntraj * n, dtype=np.float64)
    for s in range(ntraj):
        X0[s*n:(s + 1)*n] = psi0_all[s]
    return X0


@njit(fastmath=True)
def build_A_stack(Wnew, Xpnew, Xsnew, u, ntraj):
    """Block-diagonal stacked A(u) for all gate trajectories."""
    n = Wnew.shape[0]
    ns = ntraj * n
    Ablock = Wnew + u[0] * Xpnew + u[1] * Xsnew
    A = np.zeros((ns, ns), dtype=np.float64)
    for s in range(ntraj):
        i0 = s * n
        A[i0:i0+n, i0:i0+n] = Ablock
    return A


@njit(fastmath=True)
def build_B_stack(Xpnew, Xsnew, X, ntraj):
    """Stacked B = d/dU [A(U)X], shape (ntraj*n, 2)."""
    n = Xpnew.shape[0]
    ns = ntraj * n
    B = np.zeros((ns, 2), dtype=np.float64)
    for s in range(ntraj):
        i0 = s * n
        xs = X[i0:i0+n]
        B[i0:i0+n, 0] = Xpnew @ xs
        B[i0:i0+n, 1] = Xsnew @ xs
    return B


@njit(fastmath=True)
def compute_regulator_Kr_stack(Wnew, Xpnew, Xsnew, Xtraj, Ufield, kr_q, kr_r, kr_p):
    """
    Compute the PRONTO projection regulator Kr(t) for the stacked gate trajectory.

    Projection form:
        u(t) = mu(t) - Kr(t) [x(t) - alpha(t)]

    Auxiliary Riccati:
        -Pr_dot = A^T Pr + Pr A - Kr^T Rr Kr + Qr,
        Kr = Rr^{-1} B^T Pr.

    Here Qr = kr_q I, Rr = kr_r I_2, Pr(T)=kr_p I.
    Returns Kr with shape (Ntmax+1, 2, nstack).
    """
    ntime = Xtraj.shape[0]
    ns = Xtraj.shape[1]
    ntraj = ns // Wnew.shape[0]

    Pr = np.zeros((ntime, ns, ns), dtype=np.float64)
    Kr = np.zeros((ntime, 2, ns), dtype=np.float64)

    I = np.eye(ns, dtype=np.float64)
    Qr = kr_q * I
    Rr = kr_r * np.eye(2, dtype=np.float64)
    Rr_inv = (1.0 / kr_r) * np.eye(2, dtype=np.float64)

    Pr[-1] = kr_p * I

    # backward Euler for auxiliary Riccati
    for nt in range(ntime - 2, -1, -1):
        A = build_A_stack(Wnew, Xpnew, Xsnew, Ufield[nt + 1], ntraj)
        B = build_B_stack(Xpnew, Xsnew, Xtraj[nt + 1], ntraj)
        Pnext = Pr[nt + 1]

        K = Rr_inv @ (B.T @ Pnext)          # (2, ns)
        F = A.T @ Pnext + Pnext @ A - K.T @ Rr @ K + Qr
        Pcur = Pnext + deltat * F
        Pr[nt] = 0.5 * (Pcur + Pcur.T)

    for nt in range(ntime):
        B = build_B_stack(Xpnew, Xsnew, Xtraj[nt], ntraj)
        Kr[nt] = Rr_inv @ (B.T @ Pr[nt])

    return Kr


@njit(fastmath=True)
def project_with_regulator_stack(Wnew, Xpnew, Xsnew, X0, alpha, mu, Kr):
    """
    Project eta=(alpha,mu) onto the trajectory manifold using the PRONTO regulator:
        xdot = A[u(t)] x,
        u(t) = mu(t) - Kr(t) [x(t)-alpha(t)].
    """
    ntime = alpha.shape[0]
    ns = alpha.shape[1]
    ntraj = ns // Wnew.shape[0]

    Xproj = np.zeros((ntime, ns), dtype=np.float64)
    Uproj = np.zeros((ntime, 2), dtype=np.float64)
    Xproj[0] = X0.copy()

    for nt in range(ntime - 1):
        x = Xproj[nt]

        u0 = mu[nt] - Kr[nt] @ (x - alpha[nt])
        A0 = build_A_stack(Wnew, Xpnew, Xsnew, u0, ntraj)
        k1 = A0 @ x

        xpred = x + deltat * k1
        u1 = mu[nt + 1] - Kr[nt + 1] @ (xpred - alpha[nt + 1])
        A1 = build_A_stack(Wnew, Xpnew, Xsnew, u1, ntraj)
        k2 = A1 @ xpred

        Xproj[nt + 1] = x + 0.5 * deltat * (k1 + k2)
        Uproj[nt] = u0

    Uproj[-1] = mu[-1] - Kr[-1] @ (Xproj[-1] - alpha[-1])
    return Xproj, Uproj


# =========================
# Heun steps
# =========================

@njit(fastmath=True)
def heunH_step(y, nt, H, dt):
    n = y.shape[0]
    yIni = y.copy()
    k1 = np.empty(n, dtype=np.float64)
    k2 = np.empty(n, dtype=np.float64)
    tmp = np.empty(n, dtype=np.float64)

    matvec(H[nt], yIni, k1)
    for i in range(n):
        tmp[i] = yIni[i] + dt * k1[i]

    matvec(H[nt + 1], tmp, k2)
    for i in range(n):
        y[i] = yIni[i] + 0.5 * dt * (k1[i] + k2[i])


@njit(fastmath=True)
def heunXu_step(y, nt, A_k, q_k, dt):
    n = y.shape[0]
    yIni = y.copy()
    k1 = np.empty(n, dtype=np.float64)
    k2 = np.empty(n, dtype=np.float64)
    tmp = np.empty(n, dtype=np.float64)

    matvec_T(A_k[nt + 1], yIni, k1)
    for i in range(n):
        k1[i] = k1[i] - q_k[nt + 1, i]
        tmp[i] = yIni[i] - dt * k1[i]

    matvec_T(A_k[nt], tmp, k2)
    for i in range(n):
        k2[i] = k2[i] - q_k[nt, i]
        y[i] = yIni[i] - 0.5 * dt * (k1[i] + k2[i])


@njit(fastmath=True)
def heunP_step_multi(Pnext, nt, B_k, R_k, A_k, QM_k, dt):
    n = Pnext.shape[0]
    yIni = Pnext.copy()

    AT_P = np.empty((n, n), dtype=np.float64)
    P_A = np.empty((n, n), dtype=np.float64)
    matmul(A_k[nt + 1].T, yIni, AT_P)
    matmul(yIni, A_k[nt + 1], P_A)

    V1 = np.empty((2, n), dtype=np.float64)
    BT_P(B_k[nt + 1], yIni, V1)

    Rinv1 = np.empty((2, 2), dtype=np.float64)
    inv2(R_k[nt + 1], Rinv1)

    Corr1 = np.empty((n, n), dtype=np.float64)
    corr_term(V1, Rinv1, Corr1)

    k1M = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k1M[i, j] = -AT_P[i, j] - P_A[i, j] + Corr1[i, j] - QM_k[i, j]

    Ytil = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            Ytil[i, j] = yIni[i, j] - dt * k1M[i, j]

    AT_P2 = np.empty((n, n), dtype=np.float64)
    P_A2 = np.empty((n, n), dtype=np.float64)
    matmul(A_k[nt].T, Ytil, AT_P2)
    matmul(Ytil, A_k[nt], P_A2)

    V2 = np.empty((2, n), dtype=np.float64)
    BT_P(B_k[nt], Ytil, V2)

    Rinv0 = np.empty((2, 2), dtype=np.float64)
    inv2(R_k[nt], Rinv0)

    Corr2 = np.empty((n, n), dtype=np.float64)
    corr_term(V2, Rinv0, Corr2)

    k2M = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            k2M[i, j] = -AT_P2[i, j] - P_A2[i, j] + Corr2[i, j] - QM_k[i, j]

    Pout = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            Pout[i, j] = yIni[i, j] - 0.5 * dt * (k1M[i, j] + k2M[i, j])

    symmetrize(Pout)
    return Pout


@njit(fastmath=True)
def heunPm_step_multi(y, nt, A_k, B_k, K0, r_k, q_k, dt):
    n = y.shape[0]

    ABK1 = np.empty((n, n), dtype=np.float64)
    ABK2 = np.empty((n, n), dtype=np.float64)
    BK_matrix(B_k[nt + 1], K0[nt + 1], ABK1)
    BK_matrix(B_k[nt], K0[nt], ABK2)

    for i in range(n):
        for j in range(n):
            ABK1[i, j] = A_k[nt + 1, i, j] - ABK1[i, j]
            ABK2[i, j] = A_k[nt, i, j] - ABK2[i, j]

    KTr1 = np.empty(n, dtype=np.float64)
    KTr2 = np.empty(n, dtype=np.float64)
    KTr(K0[nt + 1], r_k[nt + 1], KTr1)
    KTr(K0[nt], r_k[nt], KTr2)

    yIni = y.copy()
    k1 = np.empty(n, dtype=np.float64)
    k2 = np.empty(n, dtype=np.float64)

    tmp = np.empty(n, dtype=np.float64)
    matvec_T(ABK1, yIni, tmp)
    for i in range(n):
        k1[i] = -tmp[i] + KTr1[i] - q_k[nt + 1, i]

    ytmp = np.empty(n, dtype=np.float64)
    for i in range(n):
        ytmp[i] = yIni[i] - dt * k1[i]

    matvec_T(ABK2, ytmp, tmp)
    for i in range(n):
        k2[i] = -tmp[i] + KTr2[i] - q_k[nt, i]

    for i in range(n):
        y[i] = yIni[i] - 0.5 * dt * (k1[i] + k2[i])


@njit(fastmath=True)
def heunZ_step_multi(y, nt, A_k, B_k, K0, nu0, dt):
    n = y.shape[0]
    yIni = y.copy()

    vA1 = np.empty(n, dtype=np.float64)
    vA2 = np.empty(n, dtype=np.float64)
    vBK1 = np.empty((n, n), dtype=np.float64)
    vBK2 = np.empty((n, n), dtype=np.float64)
    Bnu1 = np.empty(n, dtype=np.float64)
    Bnu2 = np.empty(n, dtype=np.float64)
    BKy1 = np.empty(n, dtype=np.float64)
    BKy2 = np.empty(n, dtype=np.float64)

    matvec(A_k[nt], yIni, vA1)
    BK_matrix(B_k[nt], K0[nt], vBK1)
    Bnu(B_k[nt], nu0[nt], Bnu1)
    matvec(vBK1, yIni, BKy1)

    k1 = np.empty(n, dtype=np.float64)
    yTmp = np.empty(n, dtype=np.float64)
    for i in range(n):
        k1[i] = vA1[i] - Bnu1[i] - BKy1[i]
        yTmp[i] = yIni[i] + dt * k1[i]

    matvec(A_k[nt + 1], yTmp, vA2)
    BK_matrix(B_k[nt + 1], K0[nt + 1], vBK2)
    Bnu(B_k[nt + 1], nu0[nt + 1], Bnu2)
    matvec(vBK2, yTmp, BKy2)

    k2 = np.empty(n, dtype=np.float64)
    for i in range(n):
        k2[i] = vA2[i] - Bnu2[i] - BKy2[i]
        y[i] = yIni[i] + 0.5 * dt * (k1[i] + k2[i])


# =========================
# PRONTO core helpers
# =========================

@njit(fastmath=True)
def build_H_timeline(Wnew, Xpnew, Xsnew, Ufield):
    n = Wnew.shape[0]
    H = np.zeros((Ntmax + 1, n, n), dtype=np.float64)
    for nt in range(Ntmax + 1):
        up = Ufield[nt, 0]
        us = Ufield[nt, 1]
        for i in range(n):
            for j in range(n):
                H[nt, i, j] = Wnew[i, j] + up * Xpnew[i, j] + us * Xsnew[i, j]
    return H


@njit(fastmath=True)
def forward_propagate(psi0, H, dt):
    n = psi0.shape[0]
    psi = np.zeros((Ntmax + 1, n), dtype=np.float64)
    psi[0] = psi0
    y = psi0.copy()
    for nt in range(Ntmax):
        heunH_step(y, nt, H, dt)
        psi[nt + 1] = y
    return psi


@njit(fastmath=True)
def forward_propagate_multi(psi0_all, H, dt):
    ntraj = psi0_all.shape[0]
    n = psi0_all.shape[1]
    psi_all = np.zeros((ntraj, Ntmax + 1, n), dtype=np.float64)

    for s in range(ntraj):
        psi_all[s, 0] = psi0_all[s]
        y = psi0_all[s].copy()
        for nt in range(Ntmax):
            heunH_step(y, nt, H, dt)
            psi_all[s, nt + 1] = y

    return psi_all


@njit(fastmath=True)
def compute_cost_multi_gate(psi_all, Ufield, R_k, target_all, Mpen, sig, dt):
    ntraj = psi_all.shape[0]
    n = psi_all.shape[2]

    vpen = np.empty(n, dtype=np.float64)

    # Phase-sensitive terminal cost:
    #   mfunc = 1/2 sum_s ||psi_s(T) - psi_target,s||^2
    # This fixes the terminal phase of each gate column.
    mfunc = 0.0
    for s in range(ntraj):
        for i in range(n):
            diff = psi_all[s, Ntmax, i] - target_all[s, i]
            mfunc += 0.5 * diff * diff

    points = np.zeros(Ntmax + 1, dtype=np.float64)
    for nt in range(Ntmax + 1):
        leak_sum = 0.0
        for s in range(ntraj):
            matvec(Mpen, psi_all[s, nt], vpen)
            leak_sum += np.dot(psi_all[s, nt], vpen)

        points[nt] = 0.5 * quadform2(Ufield[nt], R_k[nt]) + 0.5 * sig * leak_sum

    lfunc = int_simp(points, dt)
    return mfunc + lfunc, mfunc, lfunc


@njit(fastmath=True)
def build_Mproj_stack(Mproj_all):
    """Block-diagonal terminal projector for all gate trajectories."""
    ntraj = Mproj_all.shape[0]
    n = Mproj_all.shape[1]
    ns = ntraj * n
    M = np.zeros((ns, ns), dtype=np.float64)
    for s in range(ntraj):
        i0 = s * n
        M[i0:i0+n, i0:i0+n] = Mproj_all[s]
    return M


@njit(fastmath=True)
def build_Mpen_stack(Mpennew, ntraj):
    """Block-diagonal leakage projector for all gate trajectories."""
    n = Mpennew.shape[0]
    ns = ntraj * n
    M = np.zeros((ns, ns), dtype=np.float64)
    for s in range(ntraj):
        i0 = s * n
        M[i0:i0+n, i0:i0+n] = Mpennew
    return M


@njit(fastmath=True)
def compute_stacked_descent_data(Wnew, Xpnew, Xsnew, Mproj_all, Mpennew,
                                 psi_all, Ufield, R_k, sig, dt, target_all):
    """
    PRONTO Newton/quasi-Newton descent for a quantum gate in the stacked space.

    This is the correct gate formulation: both trajectories are treated as one
    stacked state X = [psi_1, psi_2], driven by the same controls
    [Omega_p, Omega_s]. No trajectory-wise averaging of controls is used.

    Returns:
      nu : (Ntmax+1, 2) control descent direction
      z_all : same shape as psi_all, stacked-state descent unstacked
      Dg : directional derivative
      modz : maximum norm of stacked z(t), used in gamma heuristic
    """
    ntraj = psi_all.shape[0]
    n = psi_all.shape[2]
    ns = ntraj * n
    ntime = Ntmax + 1

    Xtraj = stack_multi_state_traj(psi_all)

    A_k = np.zeros((ntime, ns, ns), dtype=np.float64)
    B_k = np.zeros((ntime, ns, 2), dtype=np.float64)
    q_k = np.zeros((ntime, ns), dtype=np.float64)
    r_k = np.zeros((ntime, 2), dtype=np.float64)

    Mpen_stack = build_Mpen_stack(Mpennew, ntraj)
    Q_stack = sig * Mpen_stack

    target_stack = np.zeros(ns, dtype=np.float64)
    for s in range(ntraj):
        i0 = s * n
        for j in range(n):
            target_stack[i0 + j] = target_all[s, j]

    for nt in range(ntime):
        A_k[nt] = build_A_stack(Wnew, Xpnew, Xsnew, Ufield[nt], ntraj)
        B_k[nt] = build_B_stack(Xpnew, Xsnew, Xtraj[nt], ntraj)
        q_k[nt] = sig * (Mpen_stack @ Xtraj[nt])
        r_k[nt] = R_k[nt] @ Ufield[nt]

    # Phase-sensitive terminal gradient and Hessian.
    # m = 1/2 ||X(T)-X_target||^2 gives pi = X(T)-X_target and Gamma = I.
    pi_k = Xtraj[Ntmax] - target_stack
    Gamma = np.eye(ns, dtype=np.float64)

    P = np.zeros((ntime, ns, ns), dtype=np.float64)
    p = np.zeros((ntime, ns), dtype=np.float64)
    K0 = np.zeros((ntime, 2, ns), dtype=np.float64)
    nu0 = np.zeros((ntime, 2), dtype=np.float64)

    P[Ntmax] = Gamma
    P[Ntmax] = 0.5 * (P[Ntmax] + P[Ntmax].T)
    p[Ntmax] = pi_k

    # Backward Riccati and affine equation on the stacked tangent space.
    for nt in range(Ntmax - 1, -1, -1):
        A = A_k[nt + 1]
        B = B_k[nt + 1]
        R = R_k[nt + 1]
        Rinv = np.linalg.inv(R)
        Pnext = P[nt + 1]
        pnext = p[nt + 1]
        q = q_k[nt + 1]
        r = r_k[nt + 1]

        V = B.T @ Pnext                       # (2, ns)
        K = Rinv @ V                          # (2, ns)

        # -Pdot = A^T P + P A - P B R^{-1} B^T P + Q
        Fp = A.T @ Pnext + Pnext @ A - V.T @ Rinv @ V + Q_stack
        Pcur = Pnext + dt * Fp
        Pcur = 0.5 * (Pcur + Pcur.T)
        P[nt] = Pcur

        Acl = A - B @ K
        # -pdot = (A - B K)^T p - K^T r + q
        fp = Acl.T @ pnext - K.T @ r + q
        p[nt] = pnext + dt * fp

    for nt in range(ntime):
        Rinv = np.linalg.inv(R_k[nt])
        B = B_k[nt]
        K0[nt] = Rinv @ (B.T @ P[nt])
        nu0[nt] = Rinv @ (B.T @ p[nt] + r_k[nt])

    # Forward tangent dynamics.
    z = np.zeros((ntime, ns), dtype=np.float64)
    nu = np.zeros((ntime, 2), dtype=np.float64)

    for nt in range(Ntmax):
        nu[nt] = -nu0[nt] - K0[nt] @ z[nt]
        k1 = A_k[nt] @ z[nt] + B_k[nt] @ nu[nt]

        zpred = z[nt] + dt * k1
        nupred = -nu0[nt + 1] - K0[nt + 1] @ zpred
        k2 = A_k[nt + 1] @ zpred + B_k[nt + 1] @ nupred

        z[nt + 1] = z[nt] + 0.5 * dt * (k1 + k2)

    nu[Ntmax] = -nu0[Ntmax] - K0[Ntmax] @ z[Ntmax]

    # Directional derivative.
    points = np.zeros(ntime, dtype=np.float64)
    for nt in range(ntime):
        points[nt] = np.dot(q_k[nt], z[nt]) + np.dot(r_k[nt], nu[nt])

    Dg = np.dot(pi_k, z[Ntmax]) + int_simp(points, dt)
    modz = max_norm_rows_general(z)
    z_all = unstack_multi_state_traj(z, ntraj, n)

    return nu, z_all, Dg, modz


@njit(fastmath=True)
def pronto_solve_gateX_multi_v1(Wnew, Xpnew, Xsnew, Mproj_all, Mpennew, psi0_all, target_all, Ufield, R_k,
                                sig, dt, tol, max_iter, maxls, max_riccati_tries):

    ntraj = psi0_all.shape[0]
    n = psi0_all.shape[1]

    HnewT = build_H_timeline(Wnew, Xpnew, Xsnew, Ufield)
    psi_all = forward_propagate_multi(psi0_all, HnewT, dt)

    gfunc, mfunc, lfunc = compute_cost_multi_gate(psi_all, Ufield, R_k, target_all, Mpennew, sig, dt)

    psinew_all = np.zeros_like(psi_all)
    Unew = np.zeros_like(Ufield)
    HnewTN = np.zeros_like(HnewT)

    hist_iter = np.full(max_iter, -1, dtype=np.int64)
    hist_cost = np.full(max_iter, np.nan, dtype=np.float64)
    hist_Dg = np.full(max_iter, np.nan, dtype=np.float64)
    hist_gamma = np.full(max_iter, np.nan, dtype=np.float64)
    hist_ls = np.full(max_iter, -1, dtype=np.int64)
    hist_umax = np.full(max_iter, np.nan, dtype=np.float64)
    hist_normdev = np.full(max_iter, np.nan, dtype=np.float64)

    Dg = -10.0 * tol
    nop = 0
    last_ls = 0
    c_armijo = 1.0e-4
    gamma_shrink = 0.85

    while (nop < max_iter):
        if mfunc < mfunc_tol:
            break

        nop += 1

        # Correct gate step: one stacked PRONTO Newton step for all trajectories.
        nuk_total, z_all, Dg, modz_max = compute_stacked_descent_data(
            Wnew, Xpnew, Xsnew, Mproj_all, Mpennew,
            psi_all, Ufield, R_k, sig, dt, target_all
        )

        Xtraj_stack = stack_multi_state_traj(psi_all)
        Ztraj_stack = stack_multi_state_traj(z_all)
        X0_stack = stack_initial_states(psi0_all)

        # Projection regulator on the same stacked gate trajectory.
        Kr = compute_regulator_Kr_stack(Wnew, Xpnew, Xsnew, Xtraj_stack, Ufield, kr_q, kr_r, kr_p)

        modx0 = np.sqrt(np.dot(X0_stack, X0_stack))
        gamma = 0.2
        cand = (0.6 * modx0) / (modz_max + 1e-300)
        if cand < gamma:
            gamma = cand

        accepted = False
        ls_iter = 0

        last_armijo_rhs = np.nan

        while ls_iter < maxls:
            alpha_stack = Xtraj_stack + gamma * Ztraj_stack
            mu = Ufield + gamma * nuk_total

            if (not np.all(np.isfinite(alpha_stack))) or (not np.all(np.isfinite(mu))):
                gamma *= gamma_shrink
                ls_iter += 1
                continue

            Xcand_stack, Ucand = project_with_regulator_stack(
                Wnew, Xpnew, Xsnew, X0_stack, alpha_stack, mu, Kr
            )

            if (not np.all(np.isfinite(Xcand_stack))) or (not np.all(np.isfinite(Ucand))):
                gamma *= gamma_shrink
                ls_iter += 1
                continue

            psinew_all[:] = unstack_multi_state_traj(Xcand_stack, ntraj, n)
            Unew[:] = Ucand

            HnewTN[:] = build_H_timeline(Wnew, Xpnew, Xsnew, Unew)
            if not np.all(np.isfinite(HnewTN)):
                gamma *= gamma_shrink
                ls_iter += 1
                continue

            gnew, mnew, lnew = compute_cost_multi_gate(psinew_all, Unew, R_k, target_all, Mpennew, sig, dt)
            if not np.isfinite(gnew):
                gamma *= gamma_shrink
                ls_iter += 1
                continue

            last_armijo_rhs = gfunc - c_armijo * gamma * (-Dg)

            if (Dg < 0.0) and (gnew < gfunc) and (gnew <= last_armijo_rhs):
                gfunc = gnew
                mfunc = mnew

                last_ls = ls_iter
                psi_all[:] = psinew_all
                Ufield[:] = Unew
                HnewT[:] = HnewTN
                accepted = True
                break

            gamma *= gamma_shrink
            ls_iter += 1

        if not accepted:
            break

        idx = nop - 1
        hist_iter[idx] = nop
        hist_cost[idx] = gfunc
        hist_Dg[idx] = Dg
        hist_gamma[idx] = gamma
        hist_ls[idx] = last_ls
        hist_umax[idx] = maxabs_controls(Ufield)
        hist_normdev[idx] = max_norm_deviation_multi(psi_all)


    return (
        psi_all, Ufield, gfunc, Dg, nop, last_ls,
        hist_iter, hist_cost, hist_Dg, hist_gamma, hist_ls, hist_umax, hist_normdev
    )


# =========================
# Operator construction
# =========================

def embed_complex_to_real(A):
    n = A.shape[0]
    Areal = np.zeros((2 * n, 2 * n), dtype=np.float64)
    Ar = np.real(A)
    Ai = np.imag(A)

    Areal[0:n, 0:n] = Ar
    Areal[n:2 * n, n:2 * n] = Ar
    Areal[0:n, n:2 * n] = -Ai
    Areal[n:2 * n, 0:n] = Ai
    return Areal


def build_eq49_hamiltonians(delta, Delta3, Delta4, sigma3=1.0, sigma4=1.0):
    """
    Eq. (49) in basis |1>, |2>, |3>, |4>:

    H(t) = H0 + Omega_p(t) Hp + Omega_s(t) Hs

    with couplings:
    Hp ~ (sigma3/2)|1><3| + (sigma4/2)|1><4| + h.c.
    Hs ~ (sigma3/2)|2><3| - (sigma4/2)|2><4| + h.c.
    """
    n = 4
    H0 = np.zeros((n, n), dtype=np.complex128)
    H0[1, 1] = delta
    H0[2, 2] = Delta3
    H0[3, 3] = Delta4

    Hp = np.zeros((n, n), dtype=np.complex128)
    Hp[0, 2] = 0.5 * sigma3
    Hp[2, 0] = 0.5 * sigma3
    Hp[0, 3] = 0.5 * sigma4
    Hp[3, 0] = 0.5 * sigma4

    Hs = np.zeros((n, n), dtype=np.complex128)
    Hs[1, 2] = 0.5 * sigma3
    Hs[2, 1] = 0.5 * sigma3
    Hs[1, 3] = -0.5 * sigma4
    Hs[3, 1] = -0.5 * sigma4

    return H0, Hp, Hs


def build_gateX_targets_qubit_subspace(n):
    """
    Optimize an X gate in the qubit subspace {|1>, |2>}:

      traj 0: |1> -> |2>
      traj 1: |2> -> |1>

    leakage penalty on |3>, |4>.
    """
    Xini_list = []
    Xtgt_list = []

    # |1> -> |2>
    x1 = np.zeros(n, dtype=np.complex128)
    x1[0] = 1.0 + 0.0j
    y1 = np.zeros(n, dtype=np.complex128)
    y1[1] = 1.0 + 0.0j
    Xini_list.append(x1)
    Xtgt_list.append(y1)

    # |2> -> |1>
    x2 = np.zeros(n, dtype=np.complex128)
    x2[1] = 1.0 + 0.0j
    y2 = np.zeros(n, dtype=np.complex128)
    y2[0] = 1.0 + 0.0j
    Xini_list.append(x2)
    Xtgt_list.append(y2)

    # For phase-sensitive optimization the terminal Hessian is identity.
    # Mproj_list is kept only for API compatibility with older routines.
    I = np.eye(n, dtype=np.complex128)
    Mproj_list = [I.copy(), I.copy()]

    Mpenaux = np.zeros((n, n), dtype=np.complex128)
    for k in range(2, n):
        Mpenaux[k, k] = 1.0 + 0.0j

    return Xini_list, Xtgt_list, Mproj_list, Mpenaux


def initial_controls(t, tfinal, amp_p=1.0, amp_s=1.0):
    """
    Initial guess controls requested by the user:

        sigma_p(t) = 5 t exp(-t^2)
        sigma_s(t) = -sigma_p(t)

    The amp_p and amp_s arguments are kept only for compatibility
    with the existing call initial_controls(t, tfinal, Omega0_p, Omega0_s).
    """
    U = np.zeros((t.shape[0], 2), dtype=np.float64)

    sigma_p = 5.0 * t * np.exp(-t**2)
    sigma_s = -sigma_p

    U[:, 0] = sigma_p
    U[:, 1] = sigma_s

    return U


def populations_from_real_embedded(psi, n):
    pops = np.zeros((psi.shape[0], n), dtype=np.float64)
    for k in range(n):
        pops[:, k] = psi[:, k] ** 2 + psi[:, k + n] ** 2
    return pops


def real_embedded_to_complex_state(x, n):
    return x[0:n] + 1j * x[n:2*n]


def gate_fidelity_proxy_X_from_two_trajectories(psi_all, n):
    """
    Reconstruct the effective 2x2 propagator on {|1>,|2>}
    from two propagated basis states and compare with Pauli X.

    Column 0 = evolution of initial |1>
    Column 1 = evolution of initial |2>
    """
    psi1T = real_embedded_to_complex_state(psi_all[0, -1], n)
    psi2T = real_embedded_to_complex_state(psi_all[1, -1], n)

    Uq = np.array([
        [psi1T[0], psi2T[0]],
        [psi1T[1], psi2T[1]],
    ], dtype=np.complex128)

    Xgate = np.array([
        [0.0, 1.0],
        [1.0, 0.0]
    ], dtype=np.complex128)

    trval = np.trace(np.conjugate(Xgate.T) @ Uq)
    Fgate = (np.abs(trval) ** 2) / 4.0
    return Fgate, Uq


def average_gate_fidelity_time_series_X_4level(psi_all, n):
    """
    Compute the average gate fidelity as a function of time for the final
    optimized fields in the computational subspace {|1>, |2>}.

    At each time, Ueff(t) is reconstructed from the two propagated basis
    trajectories, projected onto {|1>, |2>}:

        column 0 = evolution of initial |1>
        column 1 = evolution of initial |2>

    For d = 2, the average gate fidelity estimate is

        Favg(t) = (|Tr[X^dagger Ueff(t)]|^2 + d) / (d(d+1)).

    Since Ueff(t) may be non-unitary due to leakage to |3>, |4>, this routine
    also returns the average leakage population at each time.
    """
    ntot = psi_all.shape[1]
    d = 2

    Xgate = np.array([
        [0.0 + 0.0j, 1.0 + 0.0j],
        [1.0 + 0.0j, 0.0 + 0.0j],
    ], dtype=np.complex128)

    Favg_t = np.zeros(ntot, dtype=np.float64)
    leakage_t = np.zeros(ntot, dtype=np.float64)

    for nt in range(ntot):
        psi1_t = real_embedded_to_complex_state(psi_all[0, nt], n)
        psi2_t = real_embedded_to_complex_state(psi_all[1, nt], n)

        Ueff_t = np.array([
            [psi1_t[0], psi2_t[0]],
            [psi1_t[1], psi2_t[1]],
        ], dtype=np.complex128)

        trval = np.trace(np.conjugate(Xgate.T) @ Ueff_t)
        Favg_t[nt] = (np.abs(trval) ** 2 + d) / (d * (d + 1))

        leakage_t[nt] = 0.5 * (
            np.abs(psi1_t[2]) ** 2 + np.abs(psi1_t[3]) ** 2
            + np.abs(psi2_t[2]) ** 2 + np.abs(psi2_t[3]) ** 2
        )

    return Favg_t, leakage_t


def generate_haar_states_4level(n_samples=N_HAAR, seed=HAAR_SEED):
    """
    Generate Haar-random states in the computational subspace {|1>, |2>}
    and embed them in the 4-level Hilbert space as

        |psi> = a |1> + b |2> + 0 |3> + 0 |4>.

    Returns both the complex states and their real-embedded representation.
    """
    rng = np.random.default_rng(seed)

    z = rng.normal(size=(n_samples, 2)) + 1j * rng.normal(size=(n_samples, 2))
    z = z / np.linalg.norm(z, axis=1, keepdims=True)

    psi_c = np.zeros((n_samples, nmaxE), dtype=np.complex128)
    psi_c[:, 0] = z[:, 0]
    psi_c[:, 1] = z[:, 1]
    psi_c[:, 2] = 0.0 + 0.0j
    psi_c[:, 3] = 0.0 + 0.0j

    psi_r = np.zeros((n_samples, nvar), dtype=np.float64)
    psi_r[:, 0:nmaxE] = np.real(psi_c)
    psi_r[:, nmaxE:2*nmaxE] = np.imag(psi_c)

    return psi_c, psi_r


@njit(fastmath=True)
def _matvec_two_control(Wnew, Xpnew, Xsnew, up, us, x, y):
    n = x.shape[0]
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += (Wnew[i, j] + up * Xpnew[i, j] + us * Xsnew[i, j]) * x[j]
        y[i] = s


@njit(fastmath=True)
def haar_average_fidelity_time_series_4level(Wnew, Xpnew, Xsnew, Ufield, haar_real, Utarget):
    """
    Propagate Haar-random initial states using the final optimized fields and
    compute

        F_Haar(t) = (1/N) sum_i | <psi_i(t)| Utarget |psi_i(0)> |^2.

    The initial states have zero population in |3>, |4>. Leakage is the
    ensemble average population outside the computational subspace {|1>, |2>}.
    """
    n_samples = haar_real.shape[0]
    ntot = Ufield.shape[0]
    n = haar_real.shape[1]

    Fsum = np.zeros(ntot, dtype=np.float64)
    Lsum = np.zeros(ntot, dtype=np.float64)

    x = np.zeros(n, dtype=np.float64)
    xpred = np.zeros(n, dtype=np.float64)
    k1 = np.zeros(n, dtype=np.float64)
    k2 = np.zeros(n, dtype=np.float64)

    for m in range(n_samples):
        for i in range(n):
            x[i] = haar_real[m, i]

        # Initial qubit amplitudes a,b in {|1>,|2>}.
        a0 = haar_real[m, 0] + 1j * haar_real[m, 0 + nmaxE]
        b0 = haar_real[m, 1] + 1j * haar_real[m, 1 + nmaxE]

        # Target = Utarget @ [a0,b0].
        tgt0 = Utarget[0, 0] * a0 + Utarget[0, 1] * b0
        tgt1 = Utarget[1, 0] * a0 + Utarget[1, 1] * b0

        for nt in range(ntot):
            psi0 = x[0] + 1j * x[0 + nmaxE]
            psi1 = x[1] + 1j * x[1 + nmaxE]
            psi2 = x[2] + 1j * x[2 + nmaxE]
            psi3 = x[3] + 1j * x[3 + nmaxE]

            amp = np.conjugate(tgt0) * psi0 + np.conjugate(tgt1) * psi1
            Fsum[nt] += (amp.real * amp.real + amp.imag * amp.imag)

            Lsum[nt] += (
                psi2.real * psi2.real + psi2.imag * psi2.imag
                + psi3.real * psi3.real + psi3.imag * psi3.imag
            )

            if nt < ntot - 1:
                _matvec_two_control(Wnew, Xpnew, Xsnew, Ufield[nt, 0], Ufield[nt, 1], x, k1)
                for i in range(n):
                    xpred[i] = x[i] + deltat * k1[i]
                _matvec_two_control(Wnew, Xpnew, Xsnew, Ufield[nt + 1, 0], Ufield[nt + 1, 1], xpred, k2)
                for i in range(n):
                    x[i] = x[i] + 0.5 * deltat * (k1[i] + k2[i])

    Favg = np.zeros(ntot, dtype=np.float64)
    Lavg = np.zeros(ntot, dtype=np.float64)
    invN = 1.0 / float(n_samples)
    for nt in range(ntot):
        Favg[nt] = Fsum[nt] * invN
        Lavg[nt] = Lsum[nt] * invN

    return Favg, Lavg




# =========================
# Plotting utilities
# =========================

def configure_plot_style():
    """PRE/PRA-like matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": True,
    })


def save_combined_pdf(t, Uopt, pops_12, pops_21, out_dir=OUT_DIR):
    """
    Save one PDF with three panels:
      1) optimized Pump/Stokes fields
      2) populations for |1> -> |2>
      3) populations for |2> -> |1>
    """
    os.makedirs(out_dir, exist_ok=True)
    configure_plot_style()

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 8.0), sharex=False)

    axes[0].plot(t, Uopt[:, 0], lw=1.5, label=r"$\Omega_p(t)$")
    axes[0].plot(t, Uopt[:, 1], lw=1.5, label=r"$\Omega_s(t)$")
    axes[0].axhline(0.0, lw=0.8, color="black", alpha=0.6)
    axes[0].set_ylabel(r"$\Omega(t)$")
    axes[0].set_title(r"Optimized fields")
    axes[0].legend(loc="best", fontsize=9)

    axes[1].plot(t, pops_12[:, 0], lw=1.5, label=r"$|1\rangle$")
    axes[1].plot(t, pops_12[:, 1], lw=1.5, label=r"$|2\rangle$")
    axes[1].plot(t, pops_12[:, 2], lw=1.5, label=r"$|3\rangle$")
    axes[1].plot(t, pops_12[:, 3], lw=1.5, label=r"$|4\rangle$")
    axes[1].set_ylabel(r"Population")
    axes[1].set_ylim(-0.03, 1.05)
    axes[1].set_title(r"Trajectory $|1\rangle \rightarrow |2\rangle$")
    axes[1].legend(loc="center left", fontsize=9)

    axes[2].plot(t, pops_21[:, 0], lw=1.5, label=r"$|1\rangle$")
    axes[2].plot(t, pops_21[:, 1], lw=1.5, label=r"$|2\rangle$")
    axes[2].plot(t, pops_21[:, 2], lw=1.5, label=r"$|3\rangle$")
    axes[2].plot(t, pops_21[:, 3], lw=1.5, label=r"$|4\rangle$")
    axes[2].set_xlabel(r"$t$")
    axes[2].set_ylabel(r"Population")
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].set_title(r"Trajectory $|2\rangle \rightarrow |1\rangle$")
    axes[2].legend(loc="center left", fontsize=9)

    fig.tight_layout()

    pdf_path = os.path.join(out_dir, f"combined_{OUT_PREFIX}.pdf")
    png_path = os.path.join(out_dir, f"combined_{OUT_PREFIX}.png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved combined PDF:", pdf_path)
    print("Saved combined PNG:", png_path)

# =========================
# Main
# =========================

def main():
    print("Experimental Numba mode: Riccati + PRONTO loop + line-search compiled with @njit.")
    print("First execution includes JIT compilation overhead.")

    Xini_list, Xtgt_list, Mproj_list, Mpenaux = build_gateX_targets_qubit_subspace(nmaxE)

    ntraj = len(Xini_list)

    psi0_all = np.zeros((ntraj, nvar), dtype=np.float64)
    for s in range(ntraj):
        psi0_all[s, 0:nmaxE] = np.real(Xini_list[s])
        psi0_all[s, nmaxE:2 * nmaxE] = np.imag(Xini_list[s])

    H0, Hp, Hs = build_eq49_hamiltonians(delta, Delta3, Delta4, sigma3=sigma3, sigma4=sigma4)

    # xdot = -i H x
    Waux = -1j * H0
    Xpaux = -1j * Hp
    Xsaux = -1j * Hs

    Wnew = embed_complex_to_real(Waux)
    Xpnew = embed_complex_to_real(Xpaux)
    Xsnew = embed_complex_to_real(Xsaux)

    Mproj_all = np.zeros((ntraj, nvar, nvar), dtype=np.float64)
    for s in range(ntraj):
        Mproj_all[s] = embed_complex_to_real(Mproj_list[s])

    target_all = np.zeros((ntraj, nvar), dtype=np.float64)
    for s in range(ntraj):
        target_all[s, 0:nmaxE] = np.real(Xtgt_list[s])
        target_all[s, nmaxE:2 * nmaxE] = np.imag(Xtgt_list[s])

    Mpennew = embed_complex_to_real(Mpenaux)

    t = np.linspace(0.0, tfinal, Ntmax + 1, dtype=np.float64)
    Uini = initial_controls(t, tfinal)
    Ufield = Uini.copy()

    # diagonal control penalty matrix
    R_k = np.zeros((Ntmax + 1, 2, 2), dtype=np.float64)
    for nt in range(Ntmax + 1):
        R_k[nt, 0, 0] = 1e-3
        R_k[nt, 1, 1] = 1e-3
        R_k[nt, 0, 1] = 0.0
        R_k[nt, 1, 0] = 0.0
        if R_k[nt, 0, 0] < RMIN:
            R_k[nt, 0, 0] = RMIN
        if R_k[nt, 1, 1] < RMIN:
            R_k[nt, 1, 1] = RMIN
    t_start = time.perf_counter()
    (
        psi_all, Uopt, gfinal, Dg, iters, last_ls,
        hist_iter, hist_cost, hist_Dg, hist_gamma,
        hist_ls, hist_umax, hist_normdev
    ) = pronto_solve_gateX_multi_v1(
        Wnew, Xpnew, Xsnew, Mproj_all, Mpennew,
        psi0_all, target_all, Ufield, R_k,
        sig, deltat, tol,
        max_iter, maxls, max_riccati_tries
    )
    t_end = time.perf_counter()
    
    pops_12 = populations_from_real_embedded(psi_all[0], nmaxE)  # |1> -> ...
    pops_21 = populations_from_real_embedded(psi_all[1], nmaxE)  # |2> -> ...

    psi1_T = real_embedded_to_complex_state(psi_all[0, -1], nmaxE)
    psi2_T = real_embedded_to_complex_state(psi_all[1, -1], nmaxE)
    target1_X = np.zeros(nmaxE, dtype=np.complex128)
    target2_X = np.zeros(nmaxE, dtype=np.complex128)
    target1_X[1] = 1.0 + 0.0j
    target2_X[0] = 1.0 + 0.0j
    fid_12 = np.abs(np.vdot(target1_X, psi1_T)) ** 2
    fid_21 = np.abs(np.vdot(target2_X, psi2_T)) ** 2

    leakage_12 = np.zeros_like(t)
    leakage_21 = np.zeros_like(t)
    for k in range(2, nmaxE):
        leakage_12 += pops_12[:, k]
        leakage_21 += pops_21[:, k]





    # Leakage metrics during the gate, analogous to the 3-level code.
    leakage_mean = 0.5 * (leakage_12 + leakage_21)
    leakage_integral = np.trapezoid(leakage_mean, t)
    leakage_time_avg = leakage_integral / (t[-1] - t[0])
    leakage_max_mean = np.max(leakage_mean)
    final_leakage_avg = 0.5 * (leakage_12[-1] + leakage_21[-1])

    Fgate, Uq = gate_fidelity_proxy_X_from_two_trajectories(psi_all, nmaxE)

    # Average gate fidelity as a function of time for the optimized fields
    # 1) Previous trace/Ueff proxy, kept for comparison.
    Favg_t, leakage_gate_t = average_gate_fidelity_time_series_X_4level(psi_all, nmaxE)

    # 2) Haar Monte Carlo fidelity in the computational subspace {|1>, |2>}.
    #    Initial states have zero population in |3>, |4>.
    Utarget_X = np.array([
        [0.0 + 0.0j, 1.0 + 0.0j],
        [1.0 + 0.0j, 0.0 + 0.0j],
    ], dtype=np.complex128)
    haar_states_c, haar_states_r = generate_haar_states_4level(N_HAAR, HAAR_SEED)
    Fhaar_t, leakage_haar_t = haar_average_fidelity_time_series_4level(
        Wnew, Xpnew, Xsnew, Uopt, haar_states_r, Utarget_X
    )

    print("Done.")
    print("Fidelity |1> -> |2> =", fid_12)
    print("Fidelity |2> -> |1> =", fid_21)

    # controls

    np.savetxt("Data/OptimizedFields_Xgate_PRONTO_4L.txt", np.column_stack([t, Uopt[:, 0], Uopt[:, 1]]),
               header="t Omega_p Omega_s")
    """
    np.savetxt(
        "Data/PRONTO/AverageGateFidelity_time_Xgate_4level.txt",
        np.column_stack([t, Favg_t, leakage_gate_t]),
        header="t Favg_proxy_trace leakage_avg_t"
    )

    np.savetxt(
        "Data/PRONTO/AverageGateFidelity_Haar_time_Xgate_4level.txt",
        np.column_stack([t, Fhaar_t, leakage_haar_t]),
        header="t Favg_Haar leakage_Haar"
    )

    np.savetxt(
        "Data/PRONTO/HaarStates_Xgate_4level.txt",
        np.column_stack([
            np.real(haar_states_c[:, 0]), np.imag(haar_states_c[:, 0]),
            np.real(haar_states_c[:, 1]), np.imag(haar_states_c[:, 1]),
            np.real(haar_states_c[:, 2]), np.imag(haar_states_c[:, 2]),
            np.real(haar_states_c[:, 3]), np.imag(haar_states_c[:, 3]),
        ]),
        header="Re_a Im_a Re_b Im_b Re_c Im_c Re_d Im_d"
    )
    

    # populations traj |1> -> |2> 
    np.savetxt(
        "Data/PRONTO/Pop_traj12_all_Xgate_4L.txt",
        np.column_stack([t, pops_12]),
        header="t P1 P2 P3 P4"
        )

    np.savetxt("Data/PRONTO/Pop_traj21_all_Xgate_4L.txt", np.column_stack([t, pops_21]),
               header="t P1 P2 P3 P4")
    """

    tempo_total = t_end - t_start
    print("\n===== Simulation summary =====")
    print("Average gate fidelity proxy =", Fgate)
    print("Average gate fidelity proxy at final time =", Favg_t[-1])
    print("Average gate fidelity Haar at final time =", Fhaar_t[-1])
    print("Haar samples =", N_HAAR)
    print("Average final leakage =", final_leakage_avg)
    print("Average final Haar leakage =", leakage_haar_t[-1])

    print("\n===== Leakage during gate =====")
    print("Integrated leakage =", leakage_integral)
    print("Time-averaged leakage =", leakage_time_avg)
    print("Maximum leakage =", leakage_max_mean)

    print(f"deltat = {deltat:.6e}")
    print(f"Ntmax  = {Ntmax}")
    print(f"Total time = {tfinal}")
    print(f"Iterations = {iters}")
    print(f"Processing time = {tempo_total:.3f} s")


if __name__ == "__main__":
    main()

