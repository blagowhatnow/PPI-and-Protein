from openmm.app import PDBFile
from openmm import unit
import numpy as np
import os

def get_heavy_atom_map(pdb_path):
    """Return dict: (resSeq, atomName) -> position (nm), for heavy atoms."""
    pdb = PDBFile(pdb_path)
    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    atom_map = {}
    for atom in pdb.topology.atoms():
        if atom.element.symbol == 'H':
            continue
        key = (atom.residue.id, atom.name)
        atom_map[key] = positions[atom.index]
    return atom_map

def get_common_atoms(atom_maps):
    """Find atom keys common to all atom_maps."""
    common = set(atom_maps[0].keys())
    for am in atom_maps[1:]:
        common = common.intersection(am.keys())
    return sorted(common)

def build_coordinate_array(atom_map, atom_keys):
    """Build Nx3 numpy array of coordinates for atoms in atom_keys order."""
    coords = [atom_map[key] for key in atom_keys]
    return np.array(coords)

def kabsch_rmsd(P, Q):
    assert P.shape == Q.shape
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)

    C = np.dot(P_centered.T, Q_centered)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1, 1, d])
    U = np.dot(np.dot(V, D), Wt)

    P_rot = np.dot(P_centered, U)
    rmsd = np.sqrt(np.mean(np.sum((P_rot - Q_centered)**2, axis=1)))
    return rmsd

def compute_pairwise_rmsds(pdb_files):
    atom_maps = [get_heavy_atom_map(pdb) for pdb in pdb_files]
    common_atoms = get_common_atoms(atom_maps)
    if not common_atoms:
        raise ValueError("No common heavy atoms found across all PDBs!")

    print(f"Number of common heavy atoms: {len(common_atoms)}")

    coords = [build_coordinate_array(am, common_atoms) for am in atom_maps]

    print("Pairwise RMSDs (nm):")
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        rmsd = kabsch_rmsd(coords[i], coords[j])
        print(f"Structure {i} vs {j}: RMSD = {rmsd:.4f} nm")

if __name__ == "__main__":
    pdb_files = ["m1.pdb", "m2.pdb", "m3.pdb"]
    compute_pairwise_rmsds(pdb_files)

