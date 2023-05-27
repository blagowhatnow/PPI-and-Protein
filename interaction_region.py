#Extract the interaction region obtained using GetInterfaces.py from the original PDB
#Code written with help of ChatGPT

from Bio.PDB import PDBParser, Select
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBIO


#Residue Selector Class
class ResidueSelector(Select):
    def __init__(self, residue_ids):
        self.residue_ids = residue_ids
    
    def accept_residue(self, residue):
        if residue.get_id()[1] in self.residue_ids:
            return True
        return False

#Inputs go here
pdb_file = '5iza.pdb'
interaction1_file = 'molecule_1_constrained.pdb'
interaction2_file = 'molecule_2_constrained.pdb'

parser = PDBParser()
structure = parser.get_structure('Complete', pdb_file)
interaction1 = parser.get_structure('Interaction1', interaction1_file)
interaction2 = parser.get_structure('Interaction2', interaction2_file)

chain_ids = set()
residue_ids = set()

for model in interaction1:
    for chain in model:
        for residue in chain:
            residue_ids.add(residue.get_id()[1])

for model in interaction2:
    for chain in model:
        for residue in chain:
            residue_ids.add(residue.get_id()[1])

selector  = ResidueSelector(residue_ids)

io = PDBIO()
io.set_structure(structure)
io.save('ppi.pdb', selector)
