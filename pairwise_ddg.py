#The code here is experimental
#Might need further review

import openmm
from openmm import app, unit
import os
import numpy as np

def load_pdb_structure(pdb_file):
    """Load a PDB structure file and set up the system with OpenMM."""
    if not os.path.exists(pdb_file):
        raise ValueError(f"PDB file not found: {pdb_file}")
    pdb = app.PDBFile(pdb_file)
    # Use the proper forcefield 
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')  # Switch to a compatible forcefield
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # Solvate the system with TIP3P water model
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometer)  # Solvate with water around the system    
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    return modeller, system

def run_md_simulation(modeller, system, steps=100):
    """Run a short MD simulation to gather interaction data and compute potential energy."""
    print("Starting MD simulation...")
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Adding a reporter to write out data
    simulation.reporters.append(app.StateDataReporter('output.log', 1000, step=True, potentialEnergy=True, temperature=True))
    
    print(f"Running MD simulation for {steps} steps...")
    
    # Run the simulation for a specified number of steps
    simulation.step(steps)
    
    print("MD simulation complete. Gathering final positions...")
    # Obtain the final positions after the simulation
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    final_positions = state.getPositions(asNumpy=True)
    potential_energy = state.getPotentialEnergy()
    
    print(f"Final potential energy: {potential_energy}")
    return final_positions, potential_energy, simulation

def calculate_gibbs_free_energy(potential_energy, simulation, temperature=300 * unit.kelvin):
    """Calculate the Gibbs free energy from potential energy and entropy."""
    # Calculate the entropy (approximated by fluctuations in potential energy)
    energy_values = []
    for _ in range(10):  # Run multiple steps to collect potential energies for fluctuation estimation
        state = simulation.context.getState(getEnergy=True)
        energy_values.append(state.getPotentialEnergy())
    
    energy_values = np.array(energy_values)
    mean_energy = np.mean(energy_values)
    energy_fluctuation = np.std(energy_values)  # Standard deviation is an estimate of entropy
    
    # Entropy contribution (S = k_B * std(energy))
    k_B = unit.BOLTZMANN_CONSTANT_kB * 1e-3  # Convert to kJ/mol/K for consistency
    entropy = k_B * energy_fluctuation
    
    # Calculate Gibbs free energy: G = U + TS
    gibbs_free_energy = mean_energy + temperature * entropy
    return gibbs_free_energy

def calculate_ddg(gibbs_free_energy_1, gibbs_free_energy_2):
    """Calculate the ΔΔG (Delta Delta G) between two Gibbs free energies."""
    delta_delta_g = gibbs_free_energy_2 - gibbs_free_energy_1
    return delta_delta_g

def process_pairwise_ddg(sequences, pdb_files):
    """Main function to calculate ΔΔG for pairwise sequences."""
    
    ddg_results = {}  # Dictionary to store ΔΔG values for each pair
    
    # Loop over each pair of sequences
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1 = sequences[i]
            seq2 = sequences[j]
            
            # Prepare PDB files for each pair of sequences (here, assuming pdb_files are already prepared for each sequence)
            pdb_file_1 = pdb_files[i]  # PDB file for the first sequence
            pdb_file_2 = pdb_files[j]  # PDB file for the second sequence
            
            print(f"\nProcessing pair: {seq1[:10]}... vs {seq2[:10]}...")
            
            # Step 1: Run MD simulation for the first sequence (wildtype)
            print("Running MD simulation for the first (wildtype) system...")
            modeller_1, system_1 = load_pdb_structure(pdb_file_1)  # Wildtype PDB
            final_positions_1, potential_energy_1, simulation_1 = run_md_simulation(modeller_1, system_1)
            
            # Step 2: Run MD simulation for the second sequence (mutant)
            print("Running MD simulation for the second (mutant) system...")
            modeller_2, system_2 = load_pdb_structure(pdb_file_2)  # Mutant PDB
            final_positions_2, potential_energy_2, simulation_2 = run_md_simulation(modeller_2, system_2)

            # Step 3: Calculate Gibbs free energy for both systems
            gibbs_free_energy_1 = calculate_gibbs_free_energy(potential_energy_1, simulation_1)
            gibbs_free_energy_2 = calculate_gibbs_free_energy(potential_energy_2, simulation_2)

            # Step 4: Calculate the ΔΔG between the two simulations (mutant vs wildtype)
            delta_delta_g = calculate_ddg(gibbs_free_energy_1, gibbs_free_energy_2)
            ddg_results[(seq1[:10], seq2[:10])] = delta_delta_g  # Store ΔΔG for the sequence pair
            
            print(f"ΔΔG for {seq1[:10]}... vs {seq2[:10]}: {delta_delta_g.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    return ddg_results

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
        'p_prepared1.pdb',  # Wildtype
        'p_prepared2.pdb',  # Mutant
        'p_prepared3.pdb'
    ]

    # Call the function to process pairwise ΔΔG
    ddg_results = process_pairwise_ddg(sequences, pdb_files)

    # Print the final results
    print("\nFinal ΔΔG Results:")
    for seq_pair, ddg in ddg_results.items():
        print(f"ΔΔG for {seq_pair[0]} vs {seq_pair[1]}: {ddg.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
