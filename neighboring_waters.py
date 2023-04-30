from Bio.PDB import PDBParser, Selection,Select
from Bio.PDB import NeighborSearch
from Bio.PDB.Structure import Structure
from Bio.PDB.PDBIO import PDBIO
import Bio

def accept_residue(residue):
    """ Recognition of heteroatoms - Remove water molecules """
    res = residue.id[0]
    if res != " ": # Heteroatoms have some flags, that's why we keep only residue with id != " "
        if res != "W": # Don't take in consideration the water molecules
            return True

# Set the cutoff distance
cutoff_distance = 3.0

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

nearby_water_atoms = []
for residue in nearby_waters:
    for atom in residue.get_atoms():
        nearby_water_atoms.append(atom)
       
from Bio.PDB import Select

class ProteinSelect(Select):
    def accept_atom(self, atom):
        # Check if the atom's parent is a protein residue
        if atom.get_parent().get_resname() not in {"HOH"} and atom.get_parent().get_id()[0] == " ":
            return True
        else:
            return False

protein_select = ProteinSelect()
protein_atoms = [atom for atom in pdb.get_atoms() if protein_select.accept_atom(atom)]

class LigandWaterSelect(Select):
    def accept_atom(self, atom):
       if  atom in nearby_water_atoms or atom in protein_atoms or atom in ligand_atoms: 
           return True
       else:
            return False

# Write the selected atoms to a new PDB file
io = PDBIO()
io.set_structure(pdb)
io.save("output.pdb", LigandWaterSelect())
