# -*- coding: utf-8 -*-
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
import os

def is_het(residue):
    """
    Returns True if the residue is a heteroatom and not water.
    """
    res = residue.id[0]
    return res != " " and res != "W"

class InteractionSelect(Select):
    def __init__(self, residues_to_keep):
        self.residues_to_keep = residues_to_keep

    def accept_residue(self, residue):
        return residue in self.residues_to_keep

def extract_interaction_site_only(pdb_file, output_pdb, distance_cutoff=5.0):
    """
    Extracts only the protein residues interacting with ligand(s), excluding the ligand.

    Parameters:
    - pdb_file: Input PDB file path
    - output_pdb: Output PDB path
    - distance_cutoff: Distance threshold in Angstroms
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)
    atoms = list(structure.get_atoms())

    # Step 1: Find ligand atoms
    ligand_atoms = []
    for residue in structure.get_residues():
        if is_het(residue):
            ligand_atoms.extend(residue.get_atoms())

    if not ligand_atoms:
        raise ValueError("No ligand atoms found using is_het().")

    # Step 2: Find nearby protein residues
    ns = NeighborSearch(atoms)
    interacting_residues = set()

    for ligand_atom in ligand_atoms:
        nearby_atoms = ns.search(ligand_atom.coord, distance_cutoff)
        for atom in nearby_atoms:
            parent_residue = atom.get_parent()

            # Include only non-ligand, non-water residues
            if not is_het(parent_residue):
                interacting_residues.add(parent_residue)

    # Step 3: Write interacting residues to new PDB
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, InteractionSelect(interacting_residues))
    print("Binding site (protein only) saved to: {}".format(output_pdb))

# Example usage
if __name__ == "__main__":
    pdb_path = "test.pdb"       # Input PDB
    output_path = "binding_site_only.pdb"  # Output PDB (no ligand)
    extract_interaction_site_only(pdb_path, output_path, distance_cutoff=5.0)

