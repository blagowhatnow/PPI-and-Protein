from Bio.PDB import PDBParser, PDBIO, NeighborSearch, Select

def accept_residue(residue):
    """Identify residues to process (exclude water and heteroatoms)."""
    res = residue.id[0]
    if res == "H_" or res == " ":  # Exclude heteroatoms but include standard residues
        if residue.resname != "HOH":  # Exclude water
            return True
    return False

# Parameters
cutoff_distance = 3.0
parser = PDBParser()
pdb = parser.get_structure("prot", "2fwz.pdb")

# Collect ligand atoms
ligand_atoms = []
for model in pdb:
    for chain in model:
        for residue in chain:
            if accept_residue(residue):
                if len(list(residue.get_atoms())) > 3:  # Only process residues with sufficient atoms
                    for atom in residue.get_atoms():
                        ligand_atoms.append(atom)

# NeighborSearch to find water molecules near the ligand
ns = NeighborSearch(list(pdb.get_atoms()))
nearby_waters = []
for ligand_atom in ligand_atoms:
    neighbors = ns.search(ligand_atom.coord, cutoff_distance, level='R')
    for neighbor in neighbors:
        if neighbor.resname.strip() == "HOH":  # Check if the residue is a water molecule
            if neighbor not in nearby_waters:  # Avoid duplicates
                nearby_waters.append(neighbor)

# Collect all water atoms
nearby_water_atoms = []
for residue in nearby_waters:
    for atom in residue.get_atoms():
        nearby_water_atoms.append(atom)

# Select only protein atoms (exclude water and ligand)
class ProteinSelect(Select):
    def accept_atom(self, atom):
        parent_residue = atom.get_parent()
        if parent_residue.get_resname() != "HOH" and parent_residue.id[0] == " ":
            return True
        return False

protein_atoms = [atom for atom in pdb.get_atoms() if ProteinSelect().accept_atom(atom)]

# Custom selection for output
class LigandWaterSelect(Select):
    def accept_atom(self, atom):
        if atom in ligand_atoms or atom in nearby_water_atoms or atom in protein_atoms:
            return True
        return False

# Save selected atoms to a new PDB file
io = PDBIO()
io.set_structure(pdb)
io.save("output.pdb", LigandWaterSelect())

