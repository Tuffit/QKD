from collections import defaultdict
import numpy as np
import scipy 
from scipy.sparse import csr_matrix
from itertools import combinations

def generate_couplings(n_majorana: int, J: float = 1.0, seed: int| None = None) -> dict[tuple[int,int,int,int], float]:
    couplings=dict()
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(6 * J**2 / n_majorana**3)
    for i, j, k, l in combinations(range(n_majorana), 4):
        Jijkl = rng.normal(0, sigma)
        couplings[(i,j,k,l)] = Jijkl
    return couplings


def syk_hamiltonian(chi: list[scipy.sparse.csr_matrix], couplings: dict[tuple[int,int,int,int], float]) -> scipy.sparse.csr_matrix:
    dim = chi[0].shape[0]
    H = csr_matrix((dim, dim), dtype=complex)
    for (i, j, k, l), Jval in couplings.items():
        term = chi[i] @ chi[j] @ chi[k] @ chi[l]   
        H = H + Jval * term                        
    return H

#k is the number of eigenvalues 
def ground_state_gap(H: scipy.sparse.csr_matrix, k:int=3) -> dict:
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(H, k=k, which='SA')
    ground_state = eigenvectors[:,0]
    gap= eigenvalues[1]-eigenvalues[0]
    degenerate =  True if gap < 1e-8 else False
    return {'eigenvalues': eigenvalues,'ground_state': ground_state, 'gap':gap, 'degenerate':degenerate}