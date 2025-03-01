#This code is experimental	
#Might need further refinement

import openmm
from openmm import app, unit
import numpy as np
import os
from math import sqrt
from concurrent.futures import ProcessPoolExecutor, TimeoutError


def load_pdb_structure(pdb_file):
    """Load a PDB structure file and set up the system with OpenMM."""
    if not os.path.exists(pdb_file):
        raise ValueError(f"PDB file not found: {pdb_file}")
    
    try:
        pdb = app.PDBFile(pdb_file)
    except Exception as e:
        raise ValueError(f"Error loading PDB file {pdb_file}: {e}")
    
    forcefield = app.ForceField('amber99sb.xml', 'amber99_obc.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    return modeller, system


def calculate_distance(pos1, pos2):
    """Calculate the Euclidean distance between two 3D positions."""
    return np.linalg.norm(pos1 - pos2)


def extract_nonbonded_force(system):
    """Extract the NonbondedForce object from the system."""
    nonbonded_forces = [force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)]
    if len(nonbonded_forces) != 1:
        raise ValueError("Expected exactly one NonbondedForce in the system.")
    return nonbonded_forces[0]


def calculate_residue_interaction(modeller, system):
    """Calculate residue-residue interaction energies using OpenMM."""
    integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Minimize energy to find the stable configuration
    simulation.minimizeEnergy()

    # Extract NonbondedForce
    nonbonded_force = extract_nonbonded_force(system)

    residue_count = modeller.topology.getNumResidues()
    interaction_matrix = np.zeros((residue_count, residue_count))

    residues = list(modeller.topology.residues())
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

    for i in range(residue_count):
        for j in range(i + 1, residue_count):  # Upper triangle only
            interaction_energy = 0.0
            for atom1 in residues[i].atoms():
                for atom2 in residues[j].atoms():
                    interaction_energy += calculate_atom_interaction(
                        atom1.index, atom2.index, positions, nonbonded_force
                    )
            interaction_matrix[i, j] = interaction_matrix[j, i] = interaction_energy  # Symmetric matrix

    return interaction_matrix


def calculate_atom_interaction(index1, index2, positions, nonbonded_force):
    """Calculate the interaction energy between two atoms."""
    params1 = nonbonded_force.getParticleParameters(index1)
    params2 = nonbonded_force.getParticleParameters(index2)
    
    charge1, charge2 = params1[0], params2[0]
    epsilon1, epsilon2 = params1[1], params2[1]
    sigma1, sigma2 = params1[2], params2[2]

    pos1, pos2 = positions[index1], positions[index2]
    distance = calculate_distance(pos1, pos2)

    # Lennard-Jones potential (6-12) and Coulomb potential
    epsilon = sqrt(epsilon1 * epsilon2)
    sigma = (sigma1 + sigma2) / 2
    lj_term = 4 * epsilon * ((sigma / distance)**12 - (sigma / distance)**6)
    coulomb_term = (charge1 * charge2) / (4 * np.pi * unit.epsilon_0 * distance)

    return lj_term + coulomb_term


def calculate_potts_energy(seq1, seq2, J1, J2):
    """Calculate the Potts model energy for two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError(f"Sequences must have the same length. Got {len(seq1)} and {len(seq2)}.")
    
    energy = 0
    seq_length = len(seq1)

    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            # Skip if there's a gap ('-') in either sequence position
            if '-' in [seq1[i], seq2[i], seq1[j], seq2[j]]:
                continue

            # Interaction terms
            energy += J1[i, j] * (seq1[i] != seq2[i]) * (seq1[j] != seq2[j])
            energy += J2[i, j] * (seq1[i] != seq2[i]) * (seq1[j] != seq2[j])
    
    return energy


def validate_pdb_files(pdb_files):
    """Ensure all PDB files exist."""
    missing_files = [f for f in pdb_files if not os.path.exists(f)]
    if missing_files:
        raise ValueError(f"Missing PDB files: {', '.join(missing_files)}")


def calculate_msa_fitness(msa, pdb_files, max_workers=4):
    """Calculate fitness energies for sequences in an MSA."""
    fitness_results = {}
    num_sequences = len(msa)

    # Validate PDB files
    validate_pdb_files(pdb_files)

    # Pre-compute interaction matrices for each PDB file
    interaction_matrices = {}
    for pdb_file in pdb_files:
        if pdb_file not in interaction_matrices:
            modeller, system = load_pdb_structure(pdb_file)
            interaction_matrices[pdb_file] = calculate_residue_interaction(modeller, system)

    # Parallelize sequence pair comparisons using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i in range(num_sequences):
            seq1 = msa[i]
            pdb_file_1 = pdb_files[i]
            J1 = interaction_matrices[pdb_file_1]

            for j in range(i + 1, num_sequences):
                seq2 = msa[j]
                pdb_file_2 = pdb_files[j]
                J2 = interaction_matrices[pdb_file_2]

                futures[(i, j)] = executor.submit(calculate_potts_energy, seq1, seq2, J1, J2)
        
        # Collect results from futures
        for pair, future in futures.items():
            try:
                energy = future.result(timeout=600)  # 10 minute timeout
                fitness_results[pair] = energy
            except TimeoutError:
                print(f"Warning: Timeout error for pair {pair}")
            except Exception as e:
                print(f"Error calculating energy for pair {pair}: {e}")

    return fitness_results


# Example Sequences (Multiple Sequence Alignment) with gaps
msa = [
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGIQFVNGK",  
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGVQFVNGK",  
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGVQFVNNK",  
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGVQFVNGG",  
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGVQFVNNG",  
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGIQFVNGY",  
    "MKTAYIAKQRQISFVKSHFSRQDILDYKHGIQFVNGR",  
]

# List of paths to PDB files corresponding to each sequence
pdb_files = [
    'path/to/seq1.pdb',
    'path/to/seq2.pdb',
    'path/to/seq3.pdb',
    'path/to/seq4.pdb',
    'path/to/seq5.pdb',
    'path/to/seq6.pdb',
    'path/to/seq7.pdb',
]

# Calculate fitness energies for the MSA using the Potts model
fitness_energies = calculate_msa_fitness(msa, pdb_files)

# Print the results
for pair, energy in fitness_energies.items():
    print(f"Pair {pair} has an energy of {energy:.3f}.")

