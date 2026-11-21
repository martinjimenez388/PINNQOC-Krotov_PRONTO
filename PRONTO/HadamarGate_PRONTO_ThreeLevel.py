
# -*- coding: utf-8 -*-
"""
Q-PRONTO for a three-level system: Hadamard gate with the PRONTO regulator Kr(t).

This version implements the regulator in the sense of the PRONTO paper:

    u(t) = mu(t) - Kr(t) [x(t) - alpha(t)]

where Kr(t) is obtained from an auxiliary Riccati equation.  The Armijo
candidate is projected with this feedback law.  This is different from merely
adding a diagonal lambda to the Newton/Riccati matrices.

Gate:
    H |0> = (|0> + |1>)/sqrt(2)
    H |1> = (|0> - |1>)/sqrt(2)

State representation:
    each complex 3-level state is embedded as
        x = [Re psi0, Re psi1, Re psi2, Im psi0, Im psi1, Im psi2]
    for gate design the two basis-vector trajectories are stacked:
        X = [x_for_|0>, x_for_|1>] in R^12
"""

import time
import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def wrapper(func):
            return func
        return wrapper


# ============================================================
# Parameters
# ============================================================
Ntmax = 5000
nmaxE = 3
nvar = 2 * nmaxE          # 6
NTRAJ = 2
NSTACK = NTRAJ * nvar     # 12

tfinal = 10.0
deltat = tfinal / float(Ntmax)
pi = np.pi

# stopping criterion on terminal phase-sensitive cost
mfunc_tol = 1.0e-3
max_iter = 2000
maxls = 100

# Physical running-cost weights
sig = 0.30                # leakage penalty weight
RMIN = 1.0e-12
R_FIELD = 8.0e-4          # field fluence penalty R(t)

# PRONTO regulator design parameters, Eq. (43) style
# Qr = kr_q I, Rr = kr_r, Gamma_r = kr_p I
kr_q = 1.0
kr_r = 1.0
kr_p = 1.0

OUT_PREFIX = "Hadamard_PRONTO_reg"

# Profile timings for the Numba-compiled Riccati/Newton kernels
PROFILE_RICCATI_NUMBA = True


# ============================================================
# Numba numerical helpers
# ============================================================
@njit(fastmath=True, cache=True)
def trapz_uniform_numba(points: np.ndarray, dx: float) -> float:
    s = 0.5 * (points[0] + points[points.shape[0] - 1])
    for i in range(1, points.shape[0] - 1):
        s += points[i]
    return dx * s


@njit(fastmath=True, cache=True)
def max_norm_rows_numba(A: np.ndarray) -> float:
    m = 0.0
    for i in range(A.shape[0]):
        s = 0.0
        for j in range(A.shape[1]):
            s += A[i, j] * A[i, j]
        v = np.sqrt(s)
        if v > m:
            m = v
    return m


# ============================================================
# Complex-to-real embedding utilities
# ============================================================
def embed_complex_to_real(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    Areal = np.zeros((2 * n, 2 * n), dtype=np.float64)
    Ar = np.real(A)
    Ai = np.imag(A)
    Areal[0:n, 0:n] = Ar
    Areal[n:2 * n, n:2 * n] = Ar
    Areal[0:n, n:2 * n] = -Ai
    Areal[n:2 * n, 0:n] = Ai
    return Areal


def complex_state_to_real(psi: np.ndarray) -> np.ndarray:
    out = np.zeros(2 * psi.shape[0], dtype=np.float64)
    n = psi.shape[0]
    out[:n] = np.real(psi)
    out[n:] = np.imag(psi)
    return out


def real_state_to_complex(x: np.ndarray) -> np.ndarray:
    n = x.shape[0] // 2
    return x[:n] + 1j * x[n:]


@njit(fastmath=True, cache=True)
def stack_states(psi0_all: np.ndarray) -> np.ndarray:
    X0 = np.zeros(NSTACK, dtype=np.float64)
    for s in range(NTRAJ):
        X0[s * nvar:(s + 1) * nvar] = psi0_all[s]
    return X0


@njit(fastmath=True, cache=True)
def unstack_trajectory(Xtraj: np.ndarray) -> np.ndarray:
    psi_all = np.zeros((NTRAJ, Ntmax + 1, nvar), dtype=np.float64)
    for s in range(NTRAJ):
        psi_all[s] = Xtraj[:, s * nvar:(s + 1) * nvar]
    return psi_all


# ============================================================
# System construction
# ============================================================
def build_system_matrices():
    """Build W and Xmu in the real embedded representation."""
    Waux = np.zeros((nmaxE, nmaxE), dtype=np.complex128)
    Waux[0, 0] = -1j * 0.0
    Waux[1, 1] = -1j * 1.0
    Waux[2, 2] = -1j * 5.0

    Xmu2 = np.zeros((nmaxE, nmaxE), dtype=np.complex128)
    Xmu2[0, 1] = 0.1
    Xmu2[1, 0] = 0.1
    Xmu2[1, 2] = 0.5
    Xmu2[2, 1] = 0.5
    Xmu2[0, 2] = 0.3
    Xmu2[2, 0] = 0.3
    Xmuaux = -1j * Xmu2

    W = embed_complex_to_real(Waux)
    Xmu = embed_complex_to_real(Xmuaux)

    ket2 = np.zeros(nmaxE, dtype=np.complex128)
    ket2[2] = 1.0 + 0j
    Mpen_c = np.outer(ket2, np.conjugate(ket2))
    Mpen = embed_complex_to_real(Mpen_c)
    return W, Xmu, Mpen


@njit(fastmath=True, cache=True)
def make_A(W: np.ndarray, Xmu: np.ndarray, u: float) -> np.ndarray:
    """Stacked A(u) = I_NTRAJ \otimes (W + u Xmu)."""
    A6 = W + u * Xmu
    A = np.zeros((NSTACK, NSTACK), dtype=np.float64)
    for s in range(NTRAJ):
        i0 = s * nvar
        A[i0:i0 + nvar, i0:i0 + nvar] = A6
    return A


@njit(fastmath=True, cache=True)
def make_B(Xmu: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Stacked B = d/d u [A(u) X] = block_s Xmu x_s."""
    B = np.zeros(NSTACK, dtype=np.float64)
    for s in range(NTRAJ):
        i0 = s * nvar
        xs = X[i0:i0 + nvar]
        B[i0:i0 + nvar] = Xmu @ xs
    return B


@njit(fastmath=True, cache=True)
def make_Q_leak(Mpen: np.ndarray) -> np.ndarray:
    Q = np.zeros((NSTACK, NSTACK), dtype=np.float64)
    for s in range(NTRAJ):
        i0 = s * nvar
        Q[i0:i0 + nvar, i0:i0 + nvar] = sig * Mpen
    return Q


@njit(fastmath=True, cache=True)
def q_leak(Mpen: np.ndarray, X: np.ndarray) -> np.ndarray:
    q = np.zeros(NSTACK, dtype=np.float64)
    for s in range(NTRAJ):
        i0 = s * nvar
        xs = X[i0:i0 + nvar]
        q[i0:i0 + nvar] = sig * (Mpen @ xs)
    return q


# ============================================================
# Propagation and projection
# ============================================================
@njit(fastmath=True, cache=True)
def propagate_feasible(W: np.ndarray, Xmu: np.ndarray, X0: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Forward propagation of a feasible trajectory with prescribed input E(t)."""
    Xtraj = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    Xtraj[0] = X0.copy()

    for nt in range(Ntmax):
        x = Xtraj[nt]
        A0 = make_A(W, Xmu, E[nt])
        k1 = A0 @ x
        xpred = x + deltat * k1
        A1 = make_A(W, Xmu, E[nt + 1])
        k2 = A1 @ xpred
        Xtraj[nt + 1] = x + 0.5 * deltat * (k1 + k2)

    return Xtraj


@njit(fastmath=True, cache=True)
def project_with_regulator(
    W: np.ndarray,
    Xmu: np.ndarray,
    X0: np.ndarray,
    alpha: np.ndarray,
    mu: np.ndarray,
    Kr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projection operator P(eta) with PRONTO regulator:

        xdot = H[u] x
        u(t) = mu(t) - Kr(t) @ [x(t) - alpha(t)]

    alpha is a state curve, mu is an input curve. Kr has shape (Ntmax+1, NSTACK).
    """
    Xtraj = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    Uproj = np.zeros(Ntmax + 1, dtype=np.float64)
    Xtraj[0] = X0.copy()

    for nt in range(Ntmax):
        x = Xtraj[nt]

        u0 = mu[nt] - np.dot(Kr[nt], x - alpha[nt])
        A0 = make_A(W, Xmu, u0)
        k1 = A0 @ x

        xpred = x + deltat * k1
        u1 = mu[nt + 1] - np.dot(Kr[nt + 1], xpred - alpha[nt + 1])
        A1 = make_A(W, Xmu, u1)
        k2 = A1 @ xpred

        Xtraj[nt + 1] = x + 0.5 * deltat * (k1 + k2)
        Uproj[nt] = u0

    Uproj[Ntmax] = mu[Ntmax] - np.dot(Kr[Ntmax], Xtraj[Ntmax] - alpha[Ntmax])
    return Xtraj, Uproj


# ============================================================
# Costs
# ============================================================
@njit(fastmath=True, cache=True)
def compute_cost(Xtraj: np.ndarray, E: np.ndarray, target_stack: np.ndarray, Mpen: np.ndarray, RM: np.ndarray):
    """Total cost, terminal cost, and running cost."""
    diff = Xtraj[Ntmax] - target_stack
    mfunc = 0.5 * np.dot(diff, diff)

    points = np.zeros(Ntmax + 1, dtype=np.float64)
    for nt in range(Ntmax + 1):
        leak_sum = 0.0
        X = Xtraj[nt]
        for s in range(NTRAJ):
            i0 = s * nvar
            xs = X[i0:i0 + nvar]
            leak_sum += xs @ (Mpen @ xs)
        points[nt] = 0.5 * RM[nt] * E[nt] ** 2 + 0.5 * sig * leak_sum

    lfunc = trapz_uniform_numba(points, deltat)
    return mfunc + lfunc, mfunc, lfunc


# ============================================================
# PRONTO regulator Kr(t), Eq. (43)-style auxiliary Riccati
# ============================================================
@njit(fastmath=True, cache=True)
def compute_regulator_Kr(W: np.ndarray, Xmu: np.ndarray, Xtraj: np.ndarray, E: np.ndarray):
    """
    Compute the PRONTO projection regulator Kr(t) using the auxiliary LQR:

        -Pdot_r = A^T P_r + P_r A - K_r^T R_r K_r + Q_r
        K_r = R_r^{-1} B^T P_r

    Here we use Qr=kr_q I, Rr=kr_r, Gamma_r=kr_p I.
    For one scalar control, Kr has shape (Ntmax+1, NSTACK).
    """
    Pr = np.zeros((Ntmax + 1, NSTACK, NSTACK), dtype=np.float64)
    Kr = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)

    I = np.eye(NSTACK)
    Qr = kr_q * I
    Pr[Ntmax] = kr_p * I

    # backward Euler integration of the regulator Riccati
    for nt in range(Ntmax - 1, -1, -1):
        A = make_A(W, Xmu, E[nt + 1])
        B = make_B(Xmu, Xtraj[nt + 1]).reshape(NSTACK, 1)
        Pnext = Pr[nt + 1]

        Krow = (B.T @ Pnext) / kr_r                    # shape (1, NSTACK)
        F = A.T @ Pnext + Pnext @ A - Krow.T @ (kr_r * Krow) + Qr
        Pcur = Pnext + deltat * F
        Pcur = 0.5 * (Pcur + Pcur.T)
        Pr[nt] = Pcur

    for nt in range(Ntmax + 1):
        B = make_B(Xmu, Xtraj[nt]).reshape(NSTACK, 1)
        Kr[nt] = ((B.T @ Pr[nt]) / kr_r).ravel()

    return Kr


# ============================================================
# Newton/quasi-Newton step on tangent space
# ============================================================
@njit(fastmath=True, cache=True)
def compute_newton_step(W: np.ndarray, Xmu: np.ndarray, Mpen: np.ndarray,
                        Xtraj: np.ndarray, E: np.ndarray, target_stack: np.ndarray,
                        RM: np.ndarray):
    """
    Compute a quasi-Newton tangent step z(t), v(t) by solving the LQ problem
    with Q=leakage Hessian, R=RM, terminal Hessian Gamma=I.

    This is the standard PRONTO Newton step. The regulator Kr is used in the
    projection P(xi + sigma*zeta), not by artificially adding lambda to Q.
    """
    Aarr = np.zeros((Ntmax + 1, NSTACK, NSTACK), dtype=np.float64)
    Barr = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    qarr = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    rarr = np.zeros(Ntmax + 1, dtype=np.float64)

    Q = make_Q_leak(Mpen)
    Gamma = np.eye(NSTACK)

    for nt in range(Ntmax + 1):
        Aarr[nt] = make_A(W, Xmu, E[nt])
        Barr[nt] = make_B(Xmu, Xtraj[nt])
        qarr[nt] = q_leak(Mpen, Xtraj[nt])
        rarr[nt] = RM[nt] * E[nt]

    # terminal gradient
    pi_k = Xtraj[Ntmax] - target_stack

    # Backward Riccati for P and p
    P = np.zeros((Ntmax + 1, NSTACK, NSTACK), dtype=np.float64)
    p = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    K = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    v0 = np.zeros(Ntmax + 1, dtype=np.float64)

    P[Ntmax] = Gamma
    p[Ntmax] = pi_k

    for nt in range(Ntmax - 1, -1, -1):
        A = Aarr[nt + 1]
        B = Barr[nt + 1]
        Pnext = P[nt + 1]
        pnext = p[nt + 1]
        R = max(RM[nt + 1], RMIN)
        q = qarr[nt + 1]
        r = rarr[nt + 1]

        Krow = (B @ Pnext) / R                         # row for scalar input
        vfeed = (B @ pnext + r) / R

        # -Pdot = A^T P + P A - K^T R K + Q
        Fp = A.T @ Pnext + Pnext @ A - np.outer(Krow, Krow) * R + Q
        Pcur = Pnext + deltat * Fp
        Pcur = 0.5 * (Pcur + Pcur.T)
        P[nt] = Pcur

        Acl = A - np.outer(B, Krow)
        # -pdot = (A - B K)^T p - K^T r + q
        fp = Acl.T @ pnext - Krow * r + q
        p[nt] = pnext + deltat * fp

    for nt in range(Ntmax + 1):
        R = max(RM[nt], RMIN)
        B = Barr[nt]
        K[nt] = (B @ P[nt]) / R
        v0[nt] = (B @ p[nt] + rarr[nt]) / R

    # Forward tangent step
    z = np.zeros((Ntmax + 1, NSTACK), dtype=np.float64)
    v = np.zeros(Ntmax + 1, dtype=np.float64)

    for nt in range(Ntmax):
        v[nt] = -v0[nt] - np.dot(K[nt], z[nt])
        k1 = Aarr[nt] @ z[nt] + Barr[nt] * v[nt]

        zpred = z[nt] + deltat * k1
        vpred = -v0[nt + 1] - np.dot(K[nt + 1], zpred)
        k2 = Aarr[nt + 1] @ zpred + Barr[nt + 1] * vpred
        z[nt + 1] = z[nt] + 0.5 * deltat * (k1 + k2)

    v[Ntmax] = -v0[Ntmax] - np.dot(K[Ntmax], z[Ntmax])

    # Directional derivative Dg
    integrand = np.zeros(Ntmax + 1, dtype=np.float64)
    for nt in range(Ntmax + 1):
        integrand[nt] = np.dot(qarr[nt], z[nt]) + rarr[nt] * v[nt]
    Dg = np.dot(pi_k, z[Ntmax]) + trapz_uniform_numba(integrand, deltat)

    max_z = max_norm_rows_numba(z)
    return z, v, Dg, max_z


# ============================================================
# Solver
# ============================================================
@njit(fastmath=True, cache=True)
def pronto_solve_gate_with_regulator(W: np.ndarray, Xmu: np.ndarray, Mpen: np.ndarray,
                                     X0: np.ndarray, target_stack: np.ndarray,
                                     Efield: np.ndarray, RM: np.ndarray):
    """
    Full-Numba PRONTO loop:
      - feasible propagation
      - regulator Riccati Kr(t)
      - Newton/Riccati tangent step
      - Armijo line-search
      - projection with regulator

    This routine intentionally avoids Python timers, f-strings, and file I/O so
    it can run in Numba nopython mode.
    """
    Xtraj = propagate_feasible(W, Xmu, X0, Efield)
    gfunc, mfunc, lfunc = compute_cost(Xtraj, Efield, target_stack, Mpen, RM)

    Dg = -10.0 * mfunc_tol
    last_ls = 0
    iters_done = 0

    for nop in range(1, max_iter + 1):
        iters_done = nop

        if mfunc < mfunc_tol:
            iters_done = nop - 1
            return Xtraj, Efield, gfunc, Dg, iters_done, last_ls

        # PRONTO regulator for the current feasible trajectory.
        Kr = compute_regulator_Kr(W, Xmu, Xtraj, Efield)

        # Tangent Newton/quasi-Newton step, including Riccati.
        z, v, Dg, max_z = compute_newton_step(
            W, Xmu, Mpen, Xtraj, Efield, target_stack, RM
        )

        # Armijo step size initialization.
        norm_x0 = 0.0
        for i in range(Xtraj.shape[1]):
            norm_x0 += Xtraj[0, i] * Xtraj[0, i]
        norm_x0 = np.sqrt(norm_x0)

        sigma = 1.0
        cand = 0.6 * norm_x0 / (max_z + 1.0e-300)
        if cand < sigma:
            sigma = cand

        accepted = False

        for ls in range(maxls + 1):
            alpha = Xtraj + sigma * z
            mu = Efield + sigma * v

            # Project xi + sigma*zeta with the PRONTO regulator.
            Xcand, Ecand = project_with_regulator(W, Xmu, X0, alpha, mu, Kr)
            gcand, mcand, lcand = compute_cost(Xcand, Ecand, target_stack, Mpen, RM)

            # Dg is normally negative for descent.
            if gcand <= gfunc - 0.4 * sigma * (-Dg):
                Xtraj = Xcand
                Efield = Ecand
                gfunc = gcand
                mfunc = mcand
                last_ls = ls
                accepted = True
                break

            sigma *= 0.7

        if not accepted:
            print("WARNING: line search failed")
            print("iter", nop)
            print("mfunc", mfunc)
            print("Dg", Dg)
            return Xtraj, Efield, gfunc, Dg, nop, last_ls

    return Xtraj, Efield, gfunc, Dg, max_iter, last_ls


# ============================================================
# Diagnostics
# ============================================================
def gate_diagnostics(psi_all: np.ndarray, target_all: np.ndarray):
    psi0_T = real_state_to_complex(psi_all[0, -1])
    psi1_T = real_state_to_complex(psi_all[1, -1])

    target0 = real_state_to_complex(target_all[0])
    target1 = real_state_to_complex(target_all[1])

    amp0 = np.vdot(target0, psi0_T)
    amp1 = np.vdot(target1, psi1_T)

    fid0 = np.abs(amp0) ** 2
    fid1 = np.abs(amp1) ** 2

    print("Fidelity |0> -> (|0>+|1>)/sqrt(2) =", fid0)
    print("Fidelity |1> -> (|0>-|1>)/sqrt(2) =", fid1)


def average_gate_fidelity(psi_all: np.ndarray, Utarget: np.ndarray):
    psi0_T = real_state_to_complex(psi_all[0, -1])
    psi1_T = real_state_to_complex(psi_all[1, -1])

    Ueff = np.zeros((2, 2), dtype=np.complex128)
    Ueff[:, 0] = psi0_T[0:2]
    Ueff[:, 1] = psi1_T[0:2]

    d = 2
    tr = np.trace(np.conjugate(Utarget.T) @ Ueff)
    Favg = (np.abs(tr) ** 2 + d) / (d * (d + 1))

    leak_avg = 0.5 * (np.abs(psi0_T[2]) ** 2 + np.abs(psi1_T[2]) ** 2)
    return Favg, Ueff, leak_avg


@njit(fastmath=True, cache=True)
def average_gate_fidelity_time_series(psi_all, Utarget):
    nt = psi_all.shape[1]
    d = 2

    Favg_t = np.zeros(nt, dtype=np.float64)
    leakage_t = np.zeros(nt, dtype=np.float64)

    for k in range(nt):
        # Trajetória inicial |0>
        psi0_0 = psi_all[0, k, 0] + 1j * psi_all[0, k, 0 + nmaxE]
        psi0_1 = psi_all[0, k, 1] + 1j * psi_all[0, k, 1 + nmaxE]
        psi0_2 = psi_all[0, k, 2] + 1j * psi_all[0, k, 2 + nmaxE]

        # Trajetória inicial |1>
        psi1_0 = psi_all[1, k, 0] + 1j * psi_all[1, k, 0 + nmaxE]
        psi1_1 = psi_all[1, k, 1] + 1j * psi_all[1, k, 1 + nmaxE]
        psi1_2 = psi_all[1, k, 2] + 1j * psi_all[1, k, 2 + nmaxE]

        # Ueff(t)
        U00 = psi0_0
        U10 = psi0_1
        U01 = psi1_0
        U11 = psi1_1

        # trace(Utarget^\dagger Ueff)
        tr = (
            np.conjugate(Utarget[0, 0]) * U00
            + np.conjugate(Utarget[1, 0]) * U10
            + np.conjugate(Utarget[0, 1]) * U01
            + np.conjugate(Utarget[1, 1]) * U11
        )

        Favg_t[k] = (np.abs(tr) ** 2 + d) / (d * (d + 1))

        leakage_t[k] = 0.5 * (
            np.abs(psi0_2) ** 2 + np.abs(psi1_2) ** 2
        )

    return Favg_t, leakage_t


# ============================================================
# Main
# ============================================================
def main():
    print("Numba available =", NUMBA_AVAILABLE)
    if NUMBA_AVAILABLE:
        print("First run includes JIT compilation overhead; subsequent runs are faster.")
        print("Full Numba test: PRONTO loop, line-search, Riccati and regulator kernels are compiled with Numba.")
    W, Xmu, Mpen = build_system_matrices()

    # Initial basis states |0>, |1>
    psi0_c = np.zeros(nmaxE, dtype=np.complex128)
    psi1_c = np.zeros(nmaxE, dtype=np.complex128)
    psi0_c[0] = 1.0 + 0j
    psi1_c[1] = 1.0 + 0j

    psi0_all = np.zeros((NTRAJ, nvar), dtype=np.float64)
    psi0_all[0] = complex_state_to_real(psi0_c)
    psi0_all[1] = complex_state_to_real(psi1_c)
    X0 = stack_states(psi0_all)

    # Hadamard gate targets: H|0>=(|0>+|1>)/sqrt(2), H|1>=(|0>-|1>)/sqrt(2)
    tgt0 = np.zeros(nmaxE, dtype=np.complex128)
    tgt1 = np.zeros(nmaxE, dtype=np.complex128)
    # Hadamard gate targets
# Hadamard gate targets
    tgt0[0] = 1.0 / np.sqrt(2)
    tgt0[1] = 1.0 / np.sqrt(2)

    tgt1[0] = 1.0 / np.sqrt(2)
    tgt1[1] = -1.0 / np.sqrt(2)

    target_all = np.zeros((NTRAJ, nvar), dtype=np.float64)
    target_all[0] = complex_state_to_real(tgt0)
    target_all[1] = complex_state_to_real(tgt1)
    target_stack = stack_states(target_all)

    t = np.linspace(0.0, tfinal, Ntmax + 1, dtype=np.float64)

    # Same initial guess as the PRONTO gate example
    Efield = (pi / tfinal) * np.exp(-((t - tfinal / 2.0) ** 2) / (tfinal ** 2)) * np.cos(2.0 * pi * t)

    RM = np.maximum(np.full(Ntmax + 1, R_FIELD, dtype=np.float64), RMIN)
    t_start = time.time()
    Xtraj, Eopt, gfinal, Dg, iters, last_ls = pronto_solve_gate_with_regulator(
        W, Xmu, Mpen, X0, target_stack, Efield, RM
    )
    t_end = time.time()
    tempo_total = t_end - t_start

    psi_all = unstack_trajectory(Xtraj)

    print("Done.")


    gate_diagnostics(psi_all, target_all)

    Utarget = (1.0 / np.sqrt(2)) * np.array([
        [1.0 + 0j,  1.0 + 0j],
        [1.0 + 0j, -1.0 + 0j],
    ], dtype=np.complex128)

    Favg, Ueff, leak_avg = average_gate_fidelity(psi_all, Utarget)

    # Average gate fidelity along time for the final optimized field
    Favg_t, leakage_gate_t = average_gate_fidelity_time_series(psi_all, Utarget)

    # Save outputs
    np.savetxt(f"OptimizedField_{OUT_PREFIX}.txt", np.column_stack([t, Eopt]))
    #np.savetxt(
    #    f"AverageGateFidelity_time_{OUT_PREFIX}.txt",
    #    np.column_stack([t, Favg_t, leakage_gate_t]),
    #    header="t Favg_t leakage_avg_t"
    #)

    pop0_0 = psi_all[0, :, 0] ** 2 + psi_all[0, :, 0 + nmaxE] ** 2
    pop0_1 = psi_all[0, :, 1] ** 2 + psi_all[0, :, 1 + nmaxE] ** 2
    pop0_2 = psi_all[0, :, 2] ** 2 + psi_all[0, :, 2 + nmaxE] ** 2

    pop1_0 = psi_all[1, :, 0] ** 2 + psi_all[1, :, 0 + nmaxE] ** 2
    pop1_1 = psi_all[1, :, 1] ** 2 + psi_all[1, :, 1 + nmaxE] ** 2
    pop1_2 = psi_all[1, :, 2] ** 2 + psi_all[1, :, 2 + nmaxE] ** 2
    # ============================================================
# Leakage médio temporal
# ============================================================

# população média no nível |2>
    P2_mean = 0.5 * (pop0_2 + pop1_2)

# integral da população média
    leakage_integral = np.trapezoid(P2_mean, t)

# média temporal
    T = t[-1] - t[0]
    leakage_time_avg = leakage_integral / T

# máximo leakage durante a porta
    leakage_max = np.max(P2_mean)

    #np.savetxt(f"Data/PRONTO/Pop_traj0_all_{OUT_PREFIX}.txt", np.column_stack([t, pop0_0, pop0_1, pop0_2]), header="t P0 P1 P2")
    #np.savetxt(f"Data/PRONTO/Pop_traj1_all_{OUT_PREFIX}.txt", np.column_stack([t, pop1_0, pop1_1, pop1_2]), header="t P0 P1 P2")

    #np.savetxt(f"Data/PRONTO/Ueff_{OUT_PREFIX}.txt", Ueff.view(float).reshape(2, 4), header="Re00 Im00 Re01 Im01 / Re10 Im10 Re11 Im11")
    


    print("\n===== Simulation summary =====")
    print("Average gate fidelity =", Favg)
    print("Average gate fidelity at final time from time series =", Favg_t[-1])
    print("Average final leakage =", leak_avg)

    print("\n===== Leakage during gate =====")
    print("Integrated leakage =", leakage_integral)
    print("Time-averaged leakage =", leakage_time_avg)
    print("Maximum leakage =", leakage_max)

    print(f"deltat = {deltat:.6e}")
    print(f"Ntmax  = {Ntmax}")
    print(f"Total time = {tfinal}")
    print(f"Iterations = {iters}")
    print(f"Processing time = {tempo_total:.3f} s")

if __name__ == "__main__":
    main()