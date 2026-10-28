"""Core VQE routines: build a molecular Hamiltonian, run VQE, get reference energies."""

import numpy as onp
import pennylane as qml
from pennylane import numpy as np
from scipy.sparse.linalg import eigsh

ANGSTROM_TO_BOHR = 1.8897259886


def build_hamiltonian(symbols, bond_length, active_electrons=None, active_orbitals=None):
    # diatomic molecule along the z axis
    coords = np.array([0.0, 0.0, 0.0, 0.0, 0.0, bond_length * ANGSTROM_TO_BOHR], requires_grad=False)
    mol = qml.qchem.Molecule(symbols, coords)
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        mol, active_electrons=active_electrons, active_orbitals=active_orbitals
    )
    n_electrons = active_electrons if active_electrons is not None else mol.n_electrons
    return H, n_qubits, n_electrons


def exact_energy(H):
    # full diagonalisation (FCI within the active space) -> our ground truth
    mat = H.sparse_matrix()
    return float(eigsh(mat, k=1, which="SA")[0][0])


def make_circuit(H, n_qubits, n_electrons):
    hf_state = qml.qchem.hf_state(n_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method="backprop")
    def energy(params):
        # start from Hartree-Fock, then mix in single and double excitations (UCCSD-like)
        qml.AllSinglesDoubles(params, range(n_qubits), hf_state, singles, doubles)
        return qml.expval(H)

    n_params = len(singles) + len(doubles)
    return energy, n_params


def run_vqe(H, n_qubits, n_electrons, init_params=None, steps=200, stepsize=0.1, tol=1e-9):
    energy, n_params = make_circuit(H, n_qubits, n_electrons)

    # all zeros = plain Hartree-Fock, so this is the HF energy in the same basis
    hf_energy = float(energy(np.zeros(n_params, requires_grad=False)))

    if init_params is None:
        params = np.zeros(n_params, requires_grad=True)
    else:
        params = np.array(init_params, requires_grad=True)

    opt = qml.AdamOptimizer(stepsize=stepsize)
    history = [float(energy(params))]
    for _ in range(steps):
        params, prev = opt.step_and_cost(energy, params)
        e = float(energy(params))
        history.append(e)
        if abs(e - prev) < tol:
            break

    return {
        "energy": history[-1],
        "hf_energy": hf_energy,
        "params": onp.array(params),
        "history": history,
        "n_params": n_params,
    }
