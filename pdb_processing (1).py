import subprocess
import fastapi
import uvicorn
from openmm import *
from openmm.app import *
import pdbfixer
from fastapi import FastAPI, Query
import urllib

def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/"):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None


def prepare_protein(
    pdbcode, ignore_missing_residues=True, ignore_terminal_missing_residues=True, ph=7.0):
    """
    Use pdbfixer to prepare the protein from a PDB file. Hetero atoms such as ligands are
    removed and non-standard residues replaced. Missing atoms to existing residues are added.
    Missing residues are ignored by default, but can be included.

    Parameters
    ----------
    pdb_file: pathlib.Path or str
        PDB file containing the system to simulate.
    ignore_missing_residues: bool, optional
        If missing residues should be ignored or built.
    ignore_terminal_missing_residues: bool, optional
        If missing residues at the beginning and the end of a chain should be ignored or built.
    ph: float, optional
        pH value used to determine protonation state of residues

    Returns
    -------
    fixer: pdbfixer.pdbfixer.PDBFixer
        Prepared protein system.
    """
    download_pdb(pdbcode, "./")
    pdb_file=str(pdbcode)+'.pdb'
    fixer = pdbfixer.PDBFixer(str(pdb_file))
    fixer.removeHeterogens(keepWater=False)  # co-crystallized ligands are unknown to PDBFixer, and removing water
    fixer.findMissingResidues()  # identify missing residues, needed for identification of missing atoms

    # if missing terminal residues shall be ignored, remove them from the dictionary
    if ignore_terminal_missing_residues:
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    # if all missing residues shall be ignored ignored, clear the dictionary
    if ignore_missing_residues:
        fixer.missingResidues = {}

  
    fixer.findNonstandardResidues()  # find non-standard residue
    fixer.replaceNonstandardResidues()  # replace non-standard residues with standard one
    fixer.findMissingAtoms()  # find missing heavy atoms
    fixer.addMissingAtoms()  # add missing atoms and residues
    fixer.addMissingHydrogens(ph)  # add missing hydrogens
    return fixer


#prepare protein and build only missing non-terminal residues
prepared_protein = prepare_protein("3gfe", ignore_missing_residues=False, ph=7.0)

PDBFile.writeFile(prepared_protein.topology, prepared_protein.positions, open("p_prepared.pdb", 'w'))
import simtk.openmm as mm
import simtk.openmm.app as app
import simtk.unit as unit
import numpy as np

# Load the protein structure
pdb = app.PDBFile('p_prepared.pdb')
forcefield = app.ForceField('amber99sb.xml', 'tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)
integrator = mm.VerletIntegrator(0.002*unit.picoseconds)
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

# Define harmonic restraints on the protein backbone
restraint_force = mm.CustomExternalForce('k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
restraint_force.addGlobalParameter('k', 100.0*unit.kilocalories_per_mole/unit.angstroms**2)
restraint_force.addPerParticleParameter('x0')
restraint_force.addPerParticleParameter('y0')
restraint_force.addPerParticleParameter('z0')
for atom in pdb.topology.atoms():
    if atom.name == 'CA':
        pos = pdb.positions[atom.index]
        restraint_force.addParticle(atom.index, pos.value_in_unit(unit.nanometers))

# Add the force to the system
system.addForce(restraint_force)

# Run the simulation
simulation.minimizeEnergy(maxIterations=1000)

# Write the minimized structure to a PDB file
positions = simulation.context.getState(getPositions=True).getPositions()
app.PDBFile.writeFile(pdb.topology, positions, open('minimized.pdb', 'w'))


