#Calculate pairwise Delta Delta G-s of input sequences
#The code is experimental
#Might need further refinement

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
import sys

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
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)  # Use NoCutoff instead of CutoffPeriodic
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

def run_md_simulation(modeller, system, steps=50000):
    """Run a short MD simulation to gather interaction data and compute potential energy."""
    print("Starting MD simulation...")
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Adding a reporter to write out data
    simulation.reporters.append(app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    
    print(f"Running MD simulation for {steps} steps...")
    
    # Run the simulation for a specified number of steps
    simulation.step(steps)
    
    print("MD simulation complete. Gathering final positions...")
    # Obtain the final positions after the simulation
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = state.getPositions(asNumpy=True)
    potential_energy = state.getPotentialEnergy()
    
    print(f"Final potential energy: {potential_energy}")
    return final_positions, potential_energy

def calculate_gibbs_free_energy(potential_energy, temperature, steps=50000):
    """Estimate the Gibbs free energy using the potential energy and temperature."""
    energy_fluctuations = np.std(potential_energy)  # Standard deviation of potential energy over the simulation
    delta_s = energy_fluctuations / (temperature)  # Approximation based on fluctuations
    
    # Calculate delta H as the average potential energy
    delta_h = np.mean(potential_energy)  # Average potential energy as enthalpy approximation
    
    # Calculate Gibbs free energy: G = H - T * S
    gibbs_free_energy = delta_h - temperature * delta_s
    
    print(f"Estimated Gibbs free energy: {gibbs_free_energy} kJ/mol")
    
    return gibbs_free_energy

def calculate_delta_delta_g(potential_energy_1, potential_energy_2, temperature=300*unit.kelvin, steps=50000):
    """Calculate the delta delta G (ΔΔG) between two MD simulations."""
    # Calculate the Gibbs free energy for each state
    gibbs_free_energy_1 = calculate_gibbs_free_energy(potential_energy_1, temperature, steps)
    gibbs_free_energy_2 = calculate_gibbs_free_energy(potential_energy_2, temperature, steps)
    
    # Calculate delta delta G
    delta_delta_g = gibbs_free_energy_2 - gibbs_free_energy_1
    
    print(f"Delta Delta G (ΔΔG): {delta_delta_g} kJ/mol")
    
    return delta_delta_g

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

def extract_nonbonded_force(system):
    """Extract the NonbondedForce from the OpenMM system."""
    nonbonded_force = None
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nonbonded_force = force
            break
    if nonbonded_force is None:
        raise ValueError("No NonbondedForce found in the system.")
    return nonbonded_force

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
            interaction_energy = 0.0 * unit.kilojoule_per_mole  # Initialize interaction energy as a Quantity with correct units
            for atom1 in residues[i].atoms():
                for atom2 in residues[j].atoms():
                    # Make sure calculate_atom_interaction returns a Quantity
                    lj_term, coulomb_term = calculate_atom_interaction(
                        atom1.index, atom2.index, positions, nonbonded_force
                    )
                    interaction_energy += lj_term  # Accumulate in kJ/mol units
                    interaction_energy += coulomb_term  # Add the Coulomb term

            # Store symmetric values in the matrix
            interaction_matrix[i, j] = interaction_matrix[j, i] = interaction_energy.value_in_unit(unit.kilojoule_per_mole)
    return interaction_matrix

def calculate_atom_interaction(index1, index2, positions, nonbonded_force):
    """Calculate the interaction energy between two atoms."""
    # Get parameters for both atoms
    params1 = nonbonded_force.getParticleParameters(index1)
    params2 = nonbonded_force.getParticleParameters(index2)
    
    # Extract charge, epsilon (energy), and sigma (distance) from params1 and params2
    charge1 = params1[0].value_in_unit(unit.elementary_charge)  # Charge in Coulombs
    charge2 = params2[0].value_in_unit(unit.elementary_charge)  # Charge in Coulombs
    
    # Epsilon and sigma are extracted in their respective units
    epsilon1 = params1[1].value_in_unit(unit.nanometers)  # Epsilon in nanometers
    epsilon2 = params2[1].value_in_unit(unit.nanometers)  # Epsilon in nanometers
    sigma1 = params1[2].value_in_unit(unit.kilojoule_per_mole)  # Sigma in kJ/mol
    sigma2 = params2[2].value_in_unit(unit.kilojoule_per_mole)  # Sigma in kJ/mol

    # Positions of the atoms (in OpenMM's default unit, which is nanometers)
    pos1, pos2 = positions[index1], positions[index2]
    
    # Calculate the distance between atoms in nanometers (default unit in OpenMM)
    distance = calculate_distance(pos1, pos2)
    
    # Now, calculate the interaction using Lennard-Jones and Coulomb's Law
    epsilon = sqrt(epsilon1 * epsilon2)  # Epsilon is energy (kJ/mol)

    # Average sigma (in angstroms)
    sigma = (sigma1 + sigma2) / 2.0

    # Convert epsilon to a Quantity in kJ/mol (to match units)
    epsilon_quantity = epsilon * unit.kilojoule_per_mole

    # Lennard-Jones term (6-12 potential): energy term depends on distance in angstroms
    distance_in_angstroms = distance * 10  # Convert nanometers to angstroms for Lennard-Jones calculation
    lj_term = 4 * epsilon_quantity * ((sigma / distance_in_angstroms) ** 12 - (sigma / distance_in_angstroms) ** 6)

    # Coulomb's law constant (in proper units for kJ/mol)
    epsilon_0_value = 8.854e-3  # raw numerical value of epsilon_0 in kJ·mol⁻¹·nm⁻³·C⁻²

    # Convert distance_in_angstroms back to nanometers for the Coulomb term calculation
    distance_in_nanometers = distance_in_angstroms / 10  # Convert back from angstroms to nanometers

    # Calculate Coulomb term using distance in nanometers, and ensuring it's dimensionless
    coulomb_term_raw = (charge1 * charge2) / (4 * np.pi * epsilon_0_value * distance_in_nanometers)

    # Now we have a dimensionless Coulomb term; we convert it to kJ/mol
    coulomb_term = coulomb_term_raw * unit.kilojoule_per_mole  # Convert to kJ/mol

    # Now both terms are in kilojoule/mol, so we can safely return them
    return lj_term, coulomb_term

def process_sequences_and_pdbs(sequences, pdb_files):
    """Main function to take sequences and PDB files, and calculate MSA fitness and ΔΔG."""
    # Step 1: Align the sequences to generate MSA and alignment score matrix
    msa_alignment, alignment_matrix = align_sequences(sequences)

    # Step 2: Calculate fitness energies for the MSA using both physical and evolutionary interactions
    fitness_energies = calculate_msa_fitness(msa_alignment, pdb_files, alignment_matrix)  # Pass alignment_matrix here

    # Step 3: Run MD simulations for two sequences (mutant vs. wildtype)
    print("Running MD simulation for the first (wildtype) system...")
    modeller_1, system_1 = load_pdb_structure(pdb_files[0])  # Wildtype PDB
    final_positions_1, potential_energy_1 = run_md_simulation(modeller_1, system_1)
    
    print("Running MD simulation for the second (mutant) system...")
    modeller_2, system_2 = load_pdb_structure(pdb_files[1])  # Mutant PDB
    final_positions_2, potential_energy_2 = run_md_simulation(modeller_2, system_2)

    # Step 4: Calculate the ΔΔG between the two simulations (mutant vs wildtype)
    delta_delta_g = calculate_delta_delta_g(potential_energy_1, potential_energy_2)

    # Step 5: Print the fitness energies and ΔΔG results
    print("Fitness Energies:")
    for pair, energy in fitness_energies.items():
        print(f"Pair {pair} has an energy of {energy:.3f}.")
    
    print(f"Delta Delta G (ΔΔG) for mutant vs wildtype: {delta_delta_g:.3f} kJ/mol")

def calculate_msa_fitness(msa, pdb_files, alignment_matrix, max_workers=4):
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
                futures[(i, j)] = executor.submit(calculate_potts_energy, seq1, seq2, alignment_matrix, 
                                                  interaction_matrices[pdb_file_1], interaction_matrices[pdb_file_2])

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

if __name__ == '__main__':
    # Example usage

    # Input sequences
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAAHHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
    ]

    # List of paths to PDB files corresponding to each sequence
    pdb_files = [
        'p_prepared_1.pdb',  # Wildtype
        'p_prepared_2.pdb',  # Mutant
        'p_prepared_3.pdb'
    ]

    # Call the function
    process_sequences_and_pdbs(sequences, pdb_files)
