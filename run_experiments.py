"""
Scan bond lengths for H2 and LiH, compare VQE against Hartree-Fock and exact diagonalisation.
Writes results/*.json and figures/*.png
"""

import json
import os
import time

import numpy as onp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vqe import build_hamiltonian, exact_energy, run_vqe

CHEM_ACC = 1.6e-3  # Hartree, ~1 kcal/mol

MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "bonds": onp.round(onp.linspace(0.5, 2.5, 11), 3),
        "active_electrons": None,
        "active_orbitals": None,
    },
    "LiH": {
        "symbols": ["Li", "H"],
        "bonds": onp.round(onp.linspace(1.0, 3.0, 9), 3),
        # freeze the Li 1s core, keep 2 electrons in 3 orbitals -> 6 qubits
        "active_electrons": 2,
        "active_orbitals": 3,
    },
}

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def scan(name, cfg):
    rows = []
    params = None
    print(f"\n{name}")
    print(f"{'R (A)':>6} {'HF':>12} {'VQE':>12} {'Exact':>12} {'VQE err (mHa)':>14}")
    for r in cfg["bonds"]:
        H, n_qubits, n_el = build_hamiltonian(
            cfg["symbols"], float(r), cfg["active_electrons"], cfg["active_orbitals"]
        )
        e_exact = exact_energy(H)
        # warm start from the previous geometry, converges a lot faster
        out = run_vqe(H, n_qubits, n_el, init_params=params)
        params = out["params"]
        rows.append({
            "bond_length": float(r),
            "hf": out["hf_energy"],
            "vqe": out["energy"],
            "exact": e_exact,
            "vqe_iterations": len(out["history"]) - 1,
            "history": out["history"],
        })
        print(f"{r:6.2f} {out['hf_energy']:12.6f} {out['energy']:12.6f} {e_exact:12.6f} "
              f"{1000 * abs(out['energy'] - e_exact):14.4f}")
    print(f"qubits: {n_qubits}, circuit parameters: {out['n_params']}")
    return rows, n_qubits, out["n_params"]


def plot_molecule(name, rows, n_qubits):
    R = [x["bond_length"] for x in rows]
    hf = onp.array([x["hf"] for x in rows])
    vqe = onp.array([x["vqe"] for x in rows])
    exact = onp.array([x["exact"] for x in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(R, exact, "k-", lw=2, label="Exact (FCI)")
    ax1.plot(R, hf, "--", color="tab:red", label="Hartree-Fock")
    ax1.plot(R, vqe, "o", color="tab:blue", ms=6, label="VQE")
    ax1.set_xlabel("Bond length (Å)")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.set_title(f"{name} potential energy curve ({n_qubits} qubits)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.semilogy(R, 1000 * onp.abs(hf - exact), "--s", color="tab:red", label="Hartree-Fock")
    ax2.semilogy(R, onp.maximum(1000 * onp.abs(vqe - exact), 1e-6), "-o", color="tab:blue", label="VQE")
    ax2.axhline(1000 * CHEM_ACC, color="gray", ls=":", label="Chemical accuracy (1.6 mHa)")
    ax2.set_xlabel("Bond length (Å)")
    ax2.set_ylabel("Error vs exact (mHa)")
    ax2.set_title("Error")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(f"figures/{name.lower()}_curve.png", dpi=150)
    plt.close()


def plot_convergence(all_results):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, rows in all_results.items():
        # first geometry = cold start, the interesting one
        x = rows[0]
        err = onp.maximum(1000 * onp.abs(onp.array(x["history"]) - x["exact"]), 1e-6)
        ax.semilogy(err, label=f"{name} at {x['bond_length']} Å")
    ax.axhline(1000 * CHEM_ACC, color="gray", ls=":", label="Chemical accuracy")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Error vs exact (mHa)")
    ax.set_title("VQE convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("figures/convergence.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    t0 = time.time()
    all_results = {}
    for name, cfg in MOLECULES.items():
        rows, n_qubits, n_params = scan(name, cfg)
        all_results[name] = rows
        plot_molecule(name, rows, n_qubits)
        with open(f"results/{name.lower()}.json", "w") as f:
            json.dump({"n_qubits": n_qubits, "n_params": n_params, "scan": rows}, f, indent=1)
    plot_convergence(all_results)
    print(f"\ndone in {time.time() - t0:.0f}s")
