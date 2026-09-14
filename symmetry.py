import scipy.sparse
from functools import reduce
import numpy as np
def theta_operator(joint_chi: dict) -> scipy.sparse.csr_matrix:

    n_qubits_per_side = len(joint_chi['L']) // 2

    chi_L_odd = joint_chi['L'][1::2]
    chi_R_even = joint_chi['R'][0::2]

    U_theta = reduce(lambda a, b: a @ b, chi_L_odd + chi_R_even)
    U_theta = (2 ** n_qubits_per_side) * U_theta

    return U_theta

def apply_theta(U_theta: scipy.sparse.csr_matrix, state: np.ndarray) -> np.ndarray:
    return U_theta @ state.conj()

def theta_conjugate_operator(U_theta: scipy.sparse.csr_matrix, O: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    O_theta = U_theta @ O.conj() @ U_theta.conj().T
    return O_theta

def _sparse_max_abs(M: scipy.sparse.csr_matrix) -> float:
    M = M.tocsr()
    return float(np.abs(M.data).max()) if M.nnz else 0.0

def check_theta_symmetry(H: scipy.sparse.csr_matrix, U_theta: scipy.sparse.csr_matrix, atol: float = 1e-10) -> bool:
    H_theta = theta_conjugate_operator(U_theta, H)
    return _sparse_max_abs(H_theta - H) < atol

def ground_state_theta_eigenvalue(ground_state: np.ndarray, U_theta: scipy.sparse.csr_matrix) -> complex:
    theta_gs = apply_theta(U_theta, ground_state)
    return np.vdot(ground_state, theta_gs)

def theta_parity(O: scipy.sparse.csr_matrix, U_theta: scipy.sparse.csr_matrix, atol: float = 1e-8) -> int:

    O_theta = theta_conjugate_operator(U_theta, O)

    if _sparse_max_abs(O_theta - O) < atol:
        return 1
    if _sparse_max_abs(O_theta + O) < atol:
        return -1

    raise ValueError("O does not have a well-defined Theta parity (neither +O nor -O under Theta).")

def build_sigma_bilinear(chi_side: list, idx: tuple[int, int]) -> scipy.sparse.csr_matrix:

    a, b = idx
    return 2j * (chi_side[a] @ chi_side[b])

def build_sigma_quartic(chi_side: list, idx: tuple[int, int, int, int]) -> scipy.sparse.csr_matrix:

    a, b, c, d = idx
    return 4 * (chi_side[a] @ chi_side[b] @ chi_side[c] @ chi_side[d])