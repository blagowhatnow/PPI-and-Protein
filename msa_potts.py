#This code is experimental
#Might need further review 
import openmm
from openmm import app, unit
import numpy as np
import os
from math import sqrt
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from itertools import combinations
from math import sqrt

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
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME)  # Use NoCutoff instead of CutoffPeriodic
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
            interaction_energy = 0.0  # Initialize interaction energy as raw numeric value

            for atom1 in residues[i].atoms():
                for atom2 in residues[j].atoms():
                    # Make sure calculate_atom_interaction returns raw numeric values
                    lj_term, coulomb_term = calculate_atom_interaction(atom1.index, atom2.index, positions, nonbonded_force)
                    
                    # Convert lj_term and coulomb_term to raw values (numbers without units)
                    interaction_energy += lj_term + coulomb_term  # Both terms should be numeric now

            # Store symmetric values in the matrix
            interaction_matrix[i, j] = interaction_matrix[j, i] = interaction_energy

    return interaction_matrix


def calculate_atom_interaction(index1, index2, positions, nonbonded_force, cutoff_distance=0.6):
    """Calculate the interaction energy between two atoms with a cutoff distance, focusing only on the values."""
    # Get particle parameters from the nonbonded force (raw values)
    params1 = nonbonded_force.getParticleParameters(index1)
    params2 = nonbonded_force.getParticleParameters(index2)

    # Extract raw values (without units) for charges, sigma, and epsilon
    charge1 = params1[0].value_in_unit(unit.elementary_charge)  # Convert to raw value in elementary charge units
    charge2 = params2[0].value_in_unit(unit.elementary_charge)  # Convert to raw value in elementary charge units

    sigma1 = params1[1].value_in_unit(unit.nanometer)  # Convert to raw value in nanometers
    sigma2 = params2[1].value_in_unit(unit.nanometer)  # Convert to raw value in nanometers

    epsilon1 = params1[2].value_in_unit(unit.kilojoule_per_mole)  # Convert to raw value in kJ/mol
    epsilon2 = params2[2].value_in_unit(unit.kilojoule_per_mole)  # Convert to raw value in kJ/mol

    # Extract positions and calculate distance between the atoms (in nanometers)
    pos1, pos2 = positions[index1], positions[index2]
    distance = calculate_distance(pos1, pos2)  # Distance in nanometers

    # Apply cutoff to avoid small distances causing large energy values
    if distance < cutoff_distance:
        return 0, 0  # Skip if too close

    # Lennard-Jones potential calculation (no units attached yet)
    epsilon = np.sqrt(epsilon1 * epsilon2)
    sigma = (sigma1 + sigma2) / 2.0

    # Compute the LJ term without units (values only)
    lj_term = 4 * epsilon * ((sigma / distance) ** 12 - (sigma / distance) ** 6)

    # Coulombic potential calculation (in joules and kJ/mol, values only)
    charge1_coulombs = charge1 * 1.602e-19  # Convert to Coulombs
    charge2_coulombs = charge2 * 1.602e-19  # Convert to Coulombs

    epsilon_0_value = 8.854e-12  # Vacuum permittivity (in F/m, farad per meter)
    distance_in_meters = distance * 1e-9  # Convert nm to meters

    # Coulomb energy in joules (raw value)

    COULOMB_CONST = 8.9875517923e9  # N·m²/C²
    coulomb_energy_joules = COULOMB_CONST * (charge1_coulombs * charge2_coulombs) / distance_in_meters

    # Convert to kJ/mol
    coulomb_energy_kj_per_mol = (coulomb_energy_joules / 1000) * (6.022e23)

    # Return raw LJ and Coulomb terms
    return lj_term, coulomb_energy_kj_per_mol


def align_sequences(sequences):
    """Iteratively align sequences using Needleman-Wunsch to build an MSA and return the alignment score matrix."""
    aligned_sequences = [SeqRecord(Seq(seq), id=f"seq_{i+1}") for i, seq in enumerate(sequences)]
    
    # Initialize a pairwise aligner
    aligner = PairwiseAligner()
    aligner.mode = 'global'  # Set to global alignment

    msa = [aligned_sequences[0]]  # Initialize MSA with the first sequence
    alignment_matrix = np.zeros((len(sequences), len(sequences)))

    for i in range(1, len(sequences)):
        current_seq = aligned_sequences[i]
        
        best_score = -float("inf")
        best_alignment = None

        for aligned_seq in msa:
            # Perform the alignment
            alignment = aligner.align(current_seq.seq, aligned_seq.seq)
            
            # Select the best alignment based on score
            score = alignment.score
            if score > best_score:
                best_score = score
                best_alignment = alignment

        # Ensure that the best alignment is padded to the length of the first sequence in the MSA
        max_length = len(msa[0].seq)
        aligned_seq = Seq(str(best_alignment[0]))  # Align the sequence
        if len(aligned_seq) < max_length:
            # Pad the sequence with gaps ('-') to match the max length
            aligned_seq = aligned_seq + '-' * (max_length - len(aligned_seq))

        # Append the best-aligned sequence to the MSA
        msa.append(SeqRecord(aligned_seq, id=f"seq_{i+1}_aligned"))

        # Store the best score (alignment score) for the pair in the matrix
        for j in range(i):
            alignment_matrix[i, j] = alignment_matrix[j, i] = best_score

    return msa, alignment_matrix

def calculate_potts_energy(seq1, seq2, alignment_score, interaction_matrix_1, interaction_matrix_2, alpha=1.0, beta=1.0):
    """
    Calculate Potts-like energy combining evolutionary and physical residue interaction components.
    
    Parameters:
    - seq1, seq2: Sequences to compare (as Seq or str).
    - alignment_score: Needleman-Wunsch or MSA-derived alignment score.
    - interaction_matrix_1, interaction_matrix_2: Physical interaction matrices from OpenMM.
    - alpha: Weight for the evolutionary (alignment-based) energy term.
    - beta: Weight for the physical interaction (OpenMM) energy term.
    
    Returns:
    - energy (float): Weighted hybrid Potts-like energy.
    """
    
    # Convert sequences to strings and pad
    max_len = max(len(seq1), len(seq2))
    seq1 = str(seq1).ljust(max_len, '-')
    seq2 = str(seq2).ljust(max_len, '-')
    
    # Pad interaction matrices to match sequence length
    interaction_matrix_1 = np.pad(interaction_matrix_1, ((0, max_len - interaction_matrix_1.shape[0]), 
                                                         (0, max_len - interaction_matrix_1.shape[1])), 
                                  mode='constant', constant_values=0)
    
    interaction_matrix_2 = np.pad(interaction_matrix_2, ((0, max_len - interaction_matrix_2.shape[0]), 
                                                         (0, max_len - interaction_matrix_2.shape[1])), 
                                  mode='constant', constant_values=0)
    
    energy = 0
    seq_length = max_len

    # Iterate over residue pairs
    for i in range(seq_length):
        for j in range(i + 1, seq_length):
            if seq1[i] == '-' or seq2[i] == '-' or seq1[j] == '-' or seq2[j] == '-':
                continue  # skip gap positions

            # Evolutionary term (very simple, could be replaced with more nuanced measure)
            evo_energy = 0
            if seq1[i] != seq2[i] and seq1[j] != seq2[j]:
                evo_energy = alignment_score / seq_length

            # Physical interaction energy
            phys_energy = interaction_matrix_1[i, j] + interaction_matrix_2[i, j]

            # Weighted total energy
            energy += alpha * evo_energy + beta * phys_energy

    return energy

def calculate_msa_fitness(msa, pdb_files, alignment_matrix, alpha=1.0, beta=1.0, max_workers=4):
    """Calculate fitness energies for sequences in an MSA using both physical and evolutionary interactions."""
    fitness_results = {}
    num_sequences = len(msa)

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
            seq1 = msa[i].seq
            pdb_file_1 = pdb_files[i]
            alignment_1 = interaction_matrices[pdb_file_1]

            for j in range(i + 1, num_sequences):
                seq2 = msa[j].seq
                pdb_file_2 = pdb_files[j]
                alignment_2 = interaction_matrices[pdb_file_2]

                # Submit Potts energy calculation for each pair of sequences and their corresponding interaction matrices
                alignment_score = alignment_matrix[i, j]
                futures[(i, j)] = executor.submit(
                                  calculate_potts_energy,
                                  seq1, seq2,
                                  alignment_score,
                                  interaction_matrices[pdb_file_1],
                                  interaction_matrices[pdb_file_2],
                                  alpha,  # pass weight
                                  beta    # pass weight
                                  )


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

def process_sequences_and_pdbs(sequences, pdb_files, alpha=1.0, beta=1.0):
    """Main function to take sequences and PDB files, and calculate MSA fitness."""
    # Step 1: Align the sequences to generate MSA and alignment score matrix
    msa_alignment, alignment_matrix = align_sequences(sequences)

    # Step 2: Calculate fitness energies for the MSA using both physical and evolutionary interactions
    fitness_energies = calculate_msa_fitness(msa_alignment, pdb_files, alignment_matrix)  # Pass alignment_matrix here

    # Step 3: Print the fitness energies results in the desired format
    print("Fitness Energies:")
    for pair, energy in fitness_energies.items():
        print(f"Pair {pair} has an energy of {energy:.3f}.")


if __name__ == '__main__':
    # Example usage

    # Input sequences
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PAAAAALEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASAK",
        "PAAAAALEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFAAAAALASAK"
    ]

    # List of paths to PDB files corresponding to each sequence
    pdb_files = [
        'minimized_1.pdb',
        'minimized_2.pdb',
        'minimized_3.pdb'
    ]

    # Call the function
    process_sequences_and_pdbs(sequences, pdb_files)
