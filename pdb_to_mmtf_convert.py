import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf

# Load the PDB file into an AtomArray object

atom_array = strucio.load_structure("minimized.pdb")

file = mmtf.MMTFFile()

writer = mmtf.MMTFFile

mmtf.set_structure(file,atom_array)

mmtf_file_name = "out.mmtf"

writer.write(file,mmtf_file_name)
