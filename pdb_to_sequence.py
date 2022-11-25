from Bio.PDB import PDBParser

d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


 # Just an example input pdb
record = 'molecule_1_constrained.pdb'

 # run parser
parser = PDBParser(QUIET=True)
structure = parser.get_structure('struct', record)    

 # iterate each model, chain, and residue
 # printing out the sequence for each chain

for model in structure:
     for chain in model:
         seq = []
         for residue in chain:
             seq.append(d3to1[residue.resname])
         print('>some_header\n',''.join(seq))
