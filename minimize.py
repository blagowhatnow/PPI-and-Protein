#With suggestions from ChatGPT

import subprocess
from openmm import *
from openmm.app import *
import pdbfixer
import urllib
from openmmtools.integrators import FIREMinimizationIntegrator


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
prepared_protein = prepare_protein("7uop", ignore_missing_residues=False, ph=7.0)

PDBFile.writeFile(prepared_protein.topology, prepared_protein.positions, open("p_prepared.pdb", 'w'),keepIds=True)

#Minimize energy

from openmm.app import *
from openmm import *
from openmm.unit import *
from sys import stdout
import simtk.openmm as mm
from simtk import unit as u
from sys import stdout, exit

def OPLS_LJ(system):
    forces = {system.getForce(index).__class__.__name__: system.getForce(
        index) for index in range(system.getNumForces())}
    nonbonded_force = forces['NonbondedForce']
    lorentz = CustomNonbondedForce(
        '4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=sqrt(sigma1*sigma2); epsilon=sqrt(epsilon1*epsilon2)')
    lorentz.setNonbondedMethod(nonbonded_force.getNonbondedMethod())
    lorentz.addPerParticleParameter('sigma')
    lorentz.addPerParticleParameter('epsilon')
    lorentz.setCutoffDistance(nonbonded_force.getCutoffDistance())
    system.addForce(lorentz)

    LJset = {}
    for index in range(nonbonded_force.getNumParticles()):
        charge, sigma, epsilon = nonbonded_force.getParticleParameters(index)
        LJset[index] = (sigma, epsilon)
        lorentz.addParticle([sigma, epsilon])
        nonbonded_force.setParticleParameters(index, charge, sigma, epsilon * 0)

    for i in range(nonbonded_force.getNumExceptions()):
        (p1, p2, q, sig, eps) = nonbonded_force.getExceptionParameters(i)
        lorentz.addExclusion(p1, p2)
        if eps._value != 0.0:
            sig14 = sqrt(LJset[p1][0] * LJset[p2][0])
            eps14 = sqrt(LJset[p1][1] * LJset[p2][1]) * 0.5  # OPLS scaling for 1-4 interactions
            nonbonded_force.setExceptionParameters(i, p1, p2, q, sig14, eps14)
    return system


def Minimize(simulation,iters=0):
    simulation.minimizeEnergy(maxIterations=iters)
    position = simulation.context.getState(getPositions=True).getPositions()
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    app.PDBFile.writeFile(simulation.topology, position,
                          open('minimized.pdb', 'w'),keepIds=True)
    print('Energy at Minima is %3.3f kcal/mol' % (energy._value * KcalPerKJ))
    return simulation

pdb = app.PDBFile('p_prepared.pdb')

modeller = app.Modeller(pdb.topology, pdb.positions)
forcefield = app.ForceField('amber14/protein.ff14SB.xml')

system = forcefield.createSystem(
    modeller.topology, nonbondedMethod=app.NoCutoff,  constraints=None)
system = OPLS_LJ(system)
integrator = FIREMinimizationIntegrator(tolerance=0.0001 * u.kilojoules_per_mole / u.nanometers)
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation = Minimize(simulation,1000)
