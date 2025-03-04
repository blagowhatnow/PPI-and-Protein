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

def process_ddg_optimized(sequences, pdb_files):
    """Optimized function to calculate ΔΔG for each sequence against the wildtype."""
    # Store the Gibbs free energies for each sequence (wildtype and mutants)
    gibbs_free_energies = {}
    
    # Step 1: Run MD simulations once for each sequence (wildtype and mutants)
    for i, pdb_file in enumerate(pdb_files):
        seq = sequences[i]
        print(f"\nRunning MD simulation for sequence: {seq[:10]}...")
        
        # Load and simulate the system
        modeller, system = load_pdb_structure(pdb_file)
        final_positions, potential_energy, simulation = run_md_simulation(modeller, system)
        
        # Calculate Gibbs free energy for the sequence
        gibbs_free_energy = calculate_gibbs_free_energy(potential_energy, simulation)
        
        # Store the Gibbs free energy for this sequence
        gibbs_free_energies[seq] = gibbs_free_energy
        print(f"Calculated Gibbs free energy for {seq[:10]}: {gibbs_free_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    # Step 2: Calculate ΔΔG for each pair of sequences
    ddg_results = {}
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1 = sequences[i]
            seq2 = sequences[j]
            
            # Step 3: Calculate ΔΔG for the pair (seq1 vs seq2)
            gibbs_free_energy_1 = gibbs_free_energies[seq1]
            gibbs_free_energy_2 = gibbs_free_energies[seq2]
            
            delta_delta_g = calculate_ddg(gibbs_free_energy_1, gibbs_free_energy_2)
            ddg_results[(seq1[:10], seq2[:10])] = delta_delta_g
            
            print(f"ΔΔG for {seq1[:10]}... vs {seq2[:10]}: {delta_delta_g.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    return ddg_results

# Example usage
if __name__ == '__main__':
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAAHHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
    ]

    pdb_files = [
        'p_prepared1.pdb',  # Wildtype
        'p_prepared2.pdb',  # Mutant 1
        'p_prepared3.pdb'   # Mutant 2
    ]

    ddg_results = process_ddg_optimized(sequences, pdb_files)

    # Print final results
    print("\nFinal ΔΔG Results:")
    for seq_pair, ddg in ddg_results.items():
        print(f"ΔΔG for {seq_pair[0]} vs {seq_pair[1]}: {ddg.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
