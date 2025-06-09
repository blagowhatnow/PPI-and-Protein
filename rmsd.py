#Experimental code for computing pairwise RMSD-s from structure
# Might not be exact/correct. Left for further revision.

from openmm.app import PDBFile
from openmm import unit
import numpy as np
import os
from itertools import combinations

def get_heavy_atom_map(pdb_path, target_chain=None):
    """
    Return dict: (chain_id, residue_index, atom_name) -> position (in nm),
    only for heavy atoms. Optionally restrict to a target chain.
    """
    pdb = PDBFile(pdb_path)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    atom_map = {}

    for atom in pdb.topology.atoms():
        if atom.element.symbol == 'H':
            continue
        if target_chain and atom.residue.chain.id != target_chain:
            continue
        key = (atom.residue.chain.id, atom.residue.index, atom.name)
        atom_map[key] = positions[atom.index]

    return atom_map

def get_common_atoms(atom_maps):
    """Return atom keys common to all maps."""
    common = set(atom_maps[0].keys())
    for am in atom_maps[1:]:
        common &= set(am.keys())
    return sorted(common)

def build_coordinate_array(atom_map, atom_keys):
    """Return Nx3 numpy array of atom positions matching atom_keys."""
    return np.array([atom_map[key] for key in atom_keys])

def kabsch_rmsd(P, Q):
    """Compute RMSD using the Kabsch algorithm."""
    assert P.shape == Q.shape
    P_centered = P - P.mean(axis=0)
    Q_centered = Q - Q.mean(axis=0)

    C = np.dot(P_centered.T, Q_centered)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(V, D), Wt)

    P_rot = np.dot(P_centered, U)
    rmsd = np.sqrt(np.mean(np.sum((P_rot - Q_centered)**2, axis=1)))
    return rmsd

def compute_pairwise_rmsds(pdb_files, target_chain=None):
    print("Loading structures and extracting heavy atoms...")
    atom_maps = [get_heavy_atom_map(pdb, target_chain) for pdb in pdb_files]
    common_atoms = get_common_atoms(atom_maps)

    if not common_atoms:
        raise ValueError("No common heavy atoms found across all PDBs!")
    
    print(f"Number of common heavy atoms: {len(common_atoms)}")

    coords = [build_coordinate_array(am, common_atoms) for am in atom_maps]

    print("\nPairwise RMSDs:")
    for i, j in combinations(range(len(pdb_files)), 2):
        rmsd_nm = kabsch_rmsd(coords[i], coords[j])
        rmsd_angstrom = rmsd_nm * 10
        print(f"Structure {i} vs {j}: RMSD = {rmsd_nm:.4f} nm ({rmsd_angstrom:.2f} Å)")

if __name__ == "__main__":
    pdb_files = ["m1.pdb", "m2.pdb", "m3.pdb"]
    compute_pairwise_rmsds(pdb_files, target_chain=None)  # Set chain ID if needed
