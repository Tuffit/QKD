from functools import reduce
import numpy as np
import majorana_algebra
from mq_imbalanced import h_eta_register_pairing
import scipy.sparse
from syk_hamiltonian import ground_state_gap, syk_hamiltonian
from symmetry import _sparse_max_abs, build_sigma_bilinear, build_sigma_quartic
import warnings

def expectation(operator, state, atol= 1e-10):
    v = np.vdot(state, operator @ state)
    if abs(v.imag) > atol:
        raise ValueError("non hermitian operator")
    return float(v.real) 

def compute_xi_kappa(gs, H, H_B, sigma_A, sigma_B, atol=1e-10):
    xi = expectation(sigma_B @ H_B @ sigma_B, gs)
    sigma_B_dot = 1j * (H @ sigma_B - sigma_B @ H)

    v = np.vdot(gs, (sigma_A @ sigma_B_dot) @ gs)

    if abs(v.imag) > atol:
        warnings.warn(
            f"kappa has non-negligible imaginary part: {v.imag:.3e} "
            "(sigma_A and sigma_B may not commute)"
        )

    return {'xi': float(xi), 'kappa': float(v.real)}

def build_local_hamiltonians(
    joint_chi: dict,
    couplings_L: dict, couplings_R: dict,
    eta: float, mu: float,
    modes_A: tuple[int, ...],
    modes_B: tuple[int, ...],
    gs: np.ndarray,
) -> dict:

    n_total_modes = len(joint_chi["L"])
    dim=joint_chi["dim"]

    if not set(modes_A).isdisjoint(set(modes_B)):
        overlap= sorted(set(modes_A) & set(modes_B))
        raise ValueError(f"Modes should be disjoint: {overlap}")

    modes_A=tuple(modes_A)
    modes_B = tuple(modes_B)
    
    for idx in tuple(modes_A) + tuple(modes_B):
        if not (0 <= idx < n_total_modes):
            raise ValueError(
                f"mode index {idx} out of range [0, {n_total_modes})"
            )

    if not np.isclose(np.linalg.norm(gs),1):
        raise ValueError("gs should be normalized")

    H_L = syk_hamiltonian(joint_chi["L"], couplings_L)
    H_R = syk_hamiltonian(joint_chi["R"], couplings_R)

    H_A= (1+ eta)* H_L
    H_B=(1-eta)*H_R
    H_rest = scipy.sparse.csr_matrix((dim, dim), dtype=complex)

    for i in range(n_total_modes):
        value = 1j * mu * (joint_chi["L"][i] @ joint_chi["R"][i])
        if i in modes_A:
            H_A = H_A + value
        elif i in modes_B:
            H_B = H_B +  value
        else:
            H_rest = H_rest + value

    I= scipy.sparse.identity(dim, format='csr')



    shift_A = expectation(H_A, gs)
    shift_B= expectation(H_B, gs)
    shift_rest = expectation(H_rest, gs)

    H_A = H_A - shift_A * I
    H_B = H_B - shift_B * I
    H_rest = H_rest - shift_rest * I

    return {'H_A':H_A, 'H_B':H_B, 'H_rest':H_rest, "shift_A":float(shift_A), "shift_B":float(shift_B), "shift_rest":float(shift_rest) }


def verify_locality_condition(H, H_B, sigma_b, atol=1e-10):
    comm_total = (H @ sigma_b) - (sigma_b @ H)
    comm_local = (H_B @ sigma_b) - (sigma_b @ H_B)

    return _sparse_max_abs(comm_total - comm_local) < atol


def energy_extracted_at_theta(gs, H_B, sigma_A, sigma_B, theta):
    dim= len(gs)
    I= scipy.sparse.identity(dim, format='csr')
    rho = np.zeros((dim, dim), dtype=complex)

    for m in [-1,1]:
        P=(0.5)*(I + m *sigma_A)
        psi = P @ gs 

        U=np.cos(theta)*I - 1j*m*np.sin(theta)*sigma_B

        phi = U @ psi

        rho = rho + np.outer(phi, phi.conj() )

    energy= np.trace(rho @ H_B)

    if abs(energy.imag)>1e-10:
        raise ValueError(f"non-real energy: {energy}")
    return energy.real

def optimal_theta_sweep(gs, H_B, sigma_A, sigma_B, n_points=2000):
    theta_grid = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    energy_grid = np.empty(n_points)

    for k, theta in enumerate(theta_grid):
        energy_grid[k] = energy_extracted_at_theta(
            gs, H_B, sigma_A, sigma_B, theta
        )

    k_min = np.argmin(energy_grid)
    theta_opt = theta_grid[k_min]
    energy_min = energy_grid[k_min]
    return {'theta': float(theta_opt), 'energy':float(energy_min), 'E_B': float(-energy_min), 'theta_grid': theta_grid, 'energy_grid': energy_grid}


#careful with the sign between both papers in sin(2 theta)

def optimal_theta_closed_form(xi, kappa, sign=+1):

    if sign not in [-1,+1]:
        raise ValueError("Sign needs to be +1 or -1")

    norm = np.sqrt(xi**2 + kappa **2)

    if norm < 1e-14:
        raise ValueError("xi and kappa are ~0, theta is undifined.")

    cos_2theta = xi / norm
    sin_2theta = sign *(kappa/norm)

    theta = np.arctan2(sin_2theta, cos_2theta) / 2

    energy_predicted = (0.5)*(xi-norm)

    return {
        'theta':            float(theta),
        'energy_predicted': float(energy_predicted),
        'cos_2theta':       float(cos_2theta),
        'sin_2theta':       float(sin_2theta),
    }

def alice_injected_energy(gs, H, sigma_A):
    dim = len(gs)
    I= scipy.sparse.identity(dim, format='csr')

    E_gs = expectation(H, gs)
    E_post = 0.0

    for m in [+1,-1]:
        P=(0.5)*(I + m * sigma_A)
        psi = P @ gs
        E_post= E_post + np.vdot(psi, H@psi)

    if abs(E_post.imag) > 1e-10:
        raise ValueError(f"non-real post-measurement energy: {E_post}")

    return float(E_post.real - E_gs)

def run_qet(
    joint_chi: dict,
    couplings_L: dict,
    couplings_R: dict,
    eta: float,
    mu: float,
    modes_A: tuple[int, ...],
    modes_B: tuple[int, ...],
    idx_A: tuple[int, ...],
    idx_B: tuple[int, ...],
    sigma_A_type: str = 'quartic',
    sigma_B_type: str = 'bilinear',
    n_points: int = 2000,
    theta_sign: int = +1,
) -> dict:

    valid_types = ('bilinear', 'quartic')
    if sigma_A_type not in valid_types:
        raise ValueError(f"sigma_A_type must be one of {valid_types}, got {sigma_A_type}")
    if sigma_B_type not in valid_types:
        raise ValueError(f"sigma_B_type must be one of {valid_types}, got {sigma_B_type}")
    if sigma_A_type == sigma_B_type:
        raise ValueError(
            "sigma_A and sigma_B must have opposite Theta parity: "
            "one bilinear, one quartic"
        )

    H = h_eta_register_pairing(joint_chi, couplings_L, couplings_R, eta, mu)
    diag = ground_state_gap(H, k=2)
    gs = diag['ground_state']
    gs_energy = float(np.real(diag['eigenvalues'][0]))
    gap = float(np.real(diag['gap']))

    if sigma_A_type == 'bilinear':
        sigma_A = build_sigma_bilinear(joint_chi['L'], idx_A)
    else:
        sigma_A = build_sigma_quartic(joint_chi['L'], idx_A)

    if sigma_B_type == 'bilinear':
        sigma_B = build_sigma_bilinear(joint_chi['R'], idx_B)
    else:
        sigma_B = build_sigma_quartic(joint_chi['R'], idx_B)

    locals_ = build_local_hamiltonians(
        joint_chi, couplings_L, couplings_R, eta, mu, modes_A, modes_B, gs
    )
    H_B = locals_['H_B']

    if not verify_locality_condition(H, H_B, sigma_B):
        raise ValueError(
            "[H, sigma_B] != [H_B, sigma_B]: partition incompatible with "
            f"Bob's operator (idx_B={idx_B}, modes_B={modes_B})"
        )

    corr = compute_xi_kappa(gs, H, H_B, sigma_A, sigma_B)
    xi = corr['xi']
    kappa = corr['kappa']

    sweep = optimal_theta_sweep(gs, H_B, sigma_A, sigma_B, n_points=n_points)

    if abs(kappa) > 1e-12:
        closed = optimal_theta_closed_form(xi, kappa, sign=theta_sign)
        discrepancy = abs(sweep['energy'] - closed['energy_predicted'])
        theta_closed = closed['theta']
    else:
        discrepancy = None
        theta_closed = None

    E_A = alice_injected_energy(gs, H, sigma_A)
    E_B = sweep['E_B']
    efficiency = float(E_B / E_A) if E_A > 1e-14 else None

    return {
        'E_A': E_A,
        'E_B': E_B,
        'efficiency': efficiency,
        'theta_opt': sweep['theta'],
        'theta_closed_form': theta_closed,
        'xi': xi,
        'kappa': kappa,
        'gap': gap,
        'gs_energy': gs_energy,
        'energy_min': sweep['energy'],
        'closed_form_discrepancy': discrepancy,
        'theta_grid': sweep['theta_grid'],
        'energy_grid': sweep['energy_grid'],
    }