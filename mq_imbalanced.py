import numpy as np
import scipy
from scipy.sparse import csr_matrix
from majorana_algebra import majorana_operators
from syk_hamiltonian import syk_hamiltonian, ground_state_gap

def joint_majorana_operators(n_qubits_per_side: int) -> dict:
    n_total_qubits = 2 * n_qubits_per_side
    n_majorana_per_side = 2 * n_qubits_per_side
    majoranas = majorana_operators(n_total_qubits)
    return {
        'L': majoranas[:n_majorana_per_side],
        'R': majoranas[n_majorana_per_side:],
        'dim': 2 ** n_total_qubits,
    }

def coupling_hamiltonian(joint_chi: dict) -> scipy.sparse.csr_matrix:
    """
    H_LR = i * sum_j chi^L_j @ chi^R_j (mu=1). Termino de acoplamiento
    puro, sin Hl ni Hr -- su estado base es el TFD a beta=0.
    """
    dim = joint_chi['dim']
    pairing = csr_matrix((dim, dim), dtype=complex)
    for chi_L_j, chi_R_j in zip(joint_chi['L'], joint_chi['R']):
        pairing = pairing + chi_L_j @ chi_R_j

    return 1j * pairing

#we construct here in a block way. not in a pairing way. this is have separable info later
def h_eta_register_pairing(
    joint_chi: dict,
    couplings_L: dict[tuple[int,int,int,int], float],
    couplings_R: dict[tuple[int,int,int,int], float],
    eta: float,
    mu: float,
) -> scipy.sparse.csr_matrix:

    Hl = syk_hamiltonian(joint_chi['L'], couplings_L)
    Hr = syk_hamiltonian(joint_chi['R'], couplings_R)

    H_eta = (1+eta)*Hl + (1-eta)*Hr + mu*coupling_hamiltonian(joint_chi)

    return H_eta

def ground_state_and_gap_mq(H: scipy.sparse.csr_matrix, k: int = 50) -> dict:
    return ground_state_gap(H, k=k)


#this is just an alternative what to construct h eta. Just for double check
#This is de joint matching 
def h_eta_mq_pairing(n_pairs: int, couplings_L, couplings_R, eta: float, mu: float) -> scipy.sparse.csr_matrix:

    majoranas = majorana_operators(n_pairs)
    chi_L = majoranas[0::2]
    chi_R = majoranas[1::2]

    Hl = syk_hamiltonian(chi_L, couplings_L)
    Hr = syk_hamiltonian(chi_R, couplings_R)

    dim = 2 ** n_pairs
    identity = scipy.sparse.identity(dim, dtype=complex, format='csr')
    N = csr_matrix((dim, dim), dtype=complex)
    for chi_L_j, chi_R_j in zip(chi_L, chi_R):
        c_j = (chi_L_j + 1j * chi_R_j) / np.sqrt(2)
        c_j_dag = c_j.conj().T
        N = N + c_j_dag @ c_j

    H_LR = mu * (N - (n_pairs / 2) * identity)
    H_eta = (1+eta)*Hl + (1-eta)*Hr + H_LR

    return H_eta