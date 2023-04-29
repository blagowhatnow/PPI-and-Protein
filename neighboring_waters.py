from Bio.PDB import PDBParser, Selection,Select
from Bio.PDB import NeighborSearch
from Bio.PDB.PDBIO import PDBIO
import Bio

def accept_residue(residue):
    """ Recognition of heteroatoms - Remove water molecules """
    res = residue.id[0]
    if res != " ": # Heteroatoms have some flags, that's why we keep only residue with id != " "
        if res != "W": # Don't take in consideration the water molecules
            return True

# Set the cutoff distance
cutoff_distance = 5.0

# Create a PDB parser object
parser = PDBParser()

# Load the PDB file
pdb = PDBParser().get_structure("prot", "2fwz.pdb")
io = PDBIO()
io.set_structure(pdb)
structure=pdb

ligand_atoms=[]

for model in pdb:
  for chain in model:
     for residue in chain:
        if accept_residue(residue):
           ligand_length=len(list(residue.get_atoms()))
           if ligand_length>3:
              for atom in residue.get_atoms():
                 ligand_atoms.append(atom)


# Create a NeighborSearch object
ns = NeighborSearch(list(structure.get_atoms()))

a=[i for i in ligand_atoms]

# Find the neighboring water molecules
nearby_waters = []
for ligand_atom in a:
    neighbors = ns.search(ligand_atom.coord, cutoff_distance, level='R')
    for neighbor in neighbors:
        if neighbor.resname == "HOH": # Check if the residue is a water molecule
            nearby_waters.append(neighbor)

print(list(set(nearby_waters)))

def is_het(residue):
    res = residue.id[0]
    return res != " " and res != "W"

class ProteinSelect(Select):
    def accept_residue(self, residue):
        return not is_het(residue)

protein_atoms = []
for model in pdb:
    for chain in model:
            protein_select = ProteinSelect()
            for residue in chain:
                if protein_select.accept_residue(residue):
                    protein_atoms += residue.get_atoms()


nearby_water_atoms = []
for residue in nearby_waters:
    for atom in residue.get_atoms():
        nearby_water_atoms.append(atom)

from Bio.PDB import Select

class LigandWaterSelect(Select):
    def accept_atom(self, atom):
        if atom in protein_atoms or atom in nearby_water_atoms or atom in ligand_atoms:
            return True
        else:
            return False

# Write the selected atoms to a new PDB file
io = PDBIO()
io.set_structure(structure)
io.save("output.pdb", LigandWaterSelect())
