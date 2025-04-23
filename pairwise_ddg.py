# This code is experimental 
# Might need further review

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

def run_equilibration(modeller, system, steps=5000, temperature=300 * unit.kelvin):
    """Equilibrate the system before the main MD simulation."""
    print("Starting equilibration...")
    integrator = openmm.LangevinIntegrator(temperature, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Add reporters for energy and temperature during equilibration
    simulation.reporters.append(app.StateDataReporter('equilibration_output.log', 1000, step=True, potentialEnergy=True, temperature=True))
    
    print(f"Running equilibration for {steps} steps...")
    simulation.step(steps)  # Run equilibration
    
    # Obtain final state after equilibration
    state = simulation.context.getState(getEnergy=True)
    equilibration_energy = state.getPotentialEnergy()
    
    print("Equilibration complete.")
    print(f"Energy after equilibration: {equilibration_energy}")
    return equilibration_energy

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

def collect_energy_samples(simulation, num_samples=5000, interval=100):
    """Collect a larger number of energy samples."""
    energy_values = []
    
    for step in range(0, num_samples * interval, interval):
        simulation.step(interval)  # Advance the simulation
        state = simulation.context.getState(getEnergy=True, getKineticEnergy=True)
        total_energy = state.getPotentialEnergy() + state.getKineticEnergy()
        energy_values.append(total_energy)
    
    energy_values = np.array(energy_values)
    return energy_values

def calculate_heat_capacity(energy_values, temperature):
    """Calculate the heat capacity from energy fluctuations."""
    mean_energy = np.mean(energy_values)
    mean_energy_squared = np.mean(energy_values**2)
    
    # Energy fluctuation
    energy_fluctuation = mean_energy_squared - mean_energy**2
    
    # Heat capacity: C_V = ( <E^2> - <E>^2 ) / (k_B * T^2)
    k_B = unit.BOLTZMANN_CONSTANT_kB * 1e-3  # Convert to kJ/mol/K
    heat_capacity = energy_fluctuation / (k_B * temperature**2)
    
    return heat_capacity

def calculate_gibbs_free_energy(potential_energy, simulation, temperature=310 * unit.kelvin):
    """Calculate the Gibbs free energy from potential energy and entropy (approximated by fluctuations)."""
    
    # Collect energy values for fluctuation estimation over a larger portion of the simulation
    energy_values = collect_energy_samples(simulation, num_samples=1000, interval=100)
    
    mean_energy = np.mean(energy_values)
    
    # Calculate entropy using the refined calculation
    entropy = calculate_entropy(energy_values, temperature)
    
    # Calculate Gibbs free energy: ΔG = U - TΔS
    gibbs_free_energy = mean_energy - temperature * entropy
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
        
        # Equilibrate the system before running the MD simulation
        equilibration_energy = run_equilibration(modeller, system, steps=5000)
        
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
