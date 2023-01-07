#https://stackoverflow.com/questions/61390035/how-to-save-each-ligand-from-a-pdb-file-separately-with-bio-pdb

import os

from Bio.PDB import PDBParser, PDBIO, Select


def is_het(residue):
    res = residue.id[0]
    return res != " " and res != "W"


class ResidueSelect(Select):
    def __init__(self, chain, residue):
        self.chain = chain
        self.residue = residue

    def accept_chain(self, chain):
        return chain.id == self.chain.id

    def accept_residue(self, residue):
        """ Recognition of heteroatoms - Remove water molecules """
        return residue == self.residue and is_het(residue)


def extract_ligands(path):
    """ Extraction of the heteroatoms of .pdb files """

    for pfb_file in os.listdir(path + '/data/pdb'):
        i = 1
        if pfb_file.endswith('.pdb') and not pfb_file.startswith("lig_"):
            pdb_code = pfb_file[:-4]
            pdb = PDBParser().get_structure(pdb_code, path + '/data/pdb/' + pfb_file)
            io = PDBIO()
            io.set_structure(pdb)
            for model in pdb:
                for chain in model:
                    for residue in chain:
                        if not is_het(residue):
                            continue
                        print(f"saving {chain} {residue}")
                        io.save(f"lig_{pdb_code}_{i}.pdb", ResidueSelect(chain, residue))
                        i += 1


# Main
path = mypath

extract_ligands(path)
