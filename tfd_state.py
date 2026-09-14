from functools import reduce

import numpy as np
import majorana_algebra
import scipy.sparse
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize_scalar
from mq_imbalanced import coupling_hamiltonian

def theta_single_side(n_qubits_per_side: int) -> scipy.sparse.csr_matrix:
    chi = majorana_algebra.majorana_operators(n_qubits_per_side)
    S = chi[1::2] if n_qubits_per_side % 2 == 0 else chi[0::2]
    u = reduce(lambda a, b: a @ b, S)
    return (2 ** (n_qubits_per_side / 2)) * u

def single_side_spectrum(H_single: scipy.sparse.csr_matrix) -> dict:
    eigenvalues, eigenvectors = np.linalg.eigh(H_single.toarray())
    return {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}

def tfd_pairing(joint_chi: dict, spectrum: dict, atol: float = 1e-10) -> np.ndarray:

    H_LR = coupling_hamiltonian(joint_chi)
    _, eigenvectors = eigsh(H_LR, k=1, which='SA')
    G = eigenvectors[:, 0]

    n_qubits_per_side = len(joint_chi['L']) // 2
    d = 2 ** n_qubits_per_side
    V = spectrum['eigenvectors']

    U = theta_single_side(n_qubits_per_side)
    W = U @ V.conj()

    M = G.reshape(d, d)
    g = V.conj().T @ M @ W.conj()

    off_diag_error = np.max(np.abs(g - np.diag(np.diag(g))))
    if off_diag_error >= atol:
        raise ValueError(
            f"g is not diagonal (max off-diag = {off_diag_error:.3e} >= atol): "
            "the ground state of H_LR is not the beta=0 TFD."
        )

    diag_mag_error = np.max(np.abs(np.abs(np.diag(g)) - 1 / np.sqrt(d)))
    if diag_mag_error >= atol:
        raise ValueError(
            f"|g_nn| is not uniform (max deviation = {diag_mag_error:.3e} >= atol): "
            "the ground state of H_LR is not the beta=0 TFD."
        )

   
    TV = W * (np.sqrt(d) * np.diag(g))[np.newaxis, :]
    return TV

def construct_tfd(spectrum: dict, TV: np.ndarray, beta: float) -> np.ndarray:

    E = spectrum['eigenvalues']
    V = spectrum['eigenvectors']
    d = V.shape[0]

    psi = np.zeros(d * d, dtype=complex)
    for n in range(d):
        psi += np.exp(-beta * E[n] / 2) * np.kron(V[:, n], TV[:, n])

    return psi / np.linalg.norm(psi)

def optimal_beta(gs: np.ndarray, spectrum: dict, TV: np.ndarray,
                 beta_range: tuple = (0.001, 80.0)) -> dict:

    def neg_fidelity(beta):
        tfd = construct_tfd(spectrum, TV, beta)
        return -np.abs(np.vdot(gs, tfd)) ** 2

    result = minimize_scalar(neg_fidelity, method='bounded', bounds=beta_range)
    beta_opt = result.x
    fidelity = -result.fun

    lo, hi = beta_range
    span = hi - lo
   
    at_boundary = bool((beta_opt - lo) < 0.01 * span or (hi - beta_opt) < 0.01 * span)

    return {'beta': float(beta_opt), 'fidelity': float(fidelity), 'at_boundary': at_boundary}

def decompose_ground_state(gs: np.ndarray, spectrum: dict) -> np.ndarray:

    d = int(round(np.sqrt(len(gs))))
    n_qubits_per_side = d.bit_length() - 1
    V = spectrum['eigenvectors']

    U = theta_single_side(n_qubits_per_side)
    W = U @ V.conj()

    M = gs.reshape(d, d)
    psi = V.conj().T @ M @ W.conj()
    return psi