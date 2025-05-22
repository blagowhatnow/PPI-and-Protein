# This code is experimental and under review

import openmm
from openmm import app, unit
import os
import numpy as np

np.random.seed(42)

def load_pdb_structure(pdb_file):
    """Load a PDB structure and create an OpenMM system."""
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometer)
    
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer
        constraints=app.HBonds
    )
    
    return modeller, system


def calculate_entropy(energy_values, temperature):
    """Approximate entropy from energy fluctuations (very crude)."""
    k_B = unit.BOLTZMANN_CONSTANT_kB * 1e-3  # kJ/mol/K
    energies = np.array([e.value_in_unit(unit.kilojoule_per_mole) for e in energy_values])
    
    fluct = np.var(energies)
    temp_K = temperature.value_in_unit(unit.kelvin)
    C_v = fluct / (k_B * temp_K ** 2)
    S = C_v * np.log(temp_K)
    
    return S * unit.kilojoule_per_mole / unit.kelvin


def run_simulation(modeller, system, temperature=300 * unit.kelvin, equil_steps=5000, md_steps=100000, sample_interval=1000):
    """Run equilibration and a long MD simulation, returning energy samples and final state."""
    integrator = openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    
    print("Starting equilibration...")
    simulation.step(equil_steps)
    print("Equilibration complete. Starting production MD...")
    
    energy_samples = []
    for step in range(0, md_steps, sample_interval):
        simulation.step(sample_interval)
        state = simulation.context.getState(getEnergy=True, getKineticEnergy=True)
        energy = state.getPotentialEnergy() 
        energy_samples.append(energy)
    
    final_state = simulation.context.getState(getPositions=True, getEnergy=True)
    return simulation, energy_samples, final_state.getPotentialEnergy()


def calculate_gibbs_free_energy(energy_values, temperature=310 * unit.kelvin):
    """Compute Gibbs free energy from total energy and approximate entropy."""
    energies = np.array([e.value_in_unit(unit.kilojoule_per_mole) for e in energy_values])
    mean_energy = np.mean(energies) * unit.kilojoule_per_mole
    entropy = calculate_entropy(energy_values, temperature)
    gibbs_free_energy = mean_energy - temperature * entropy
    return gibbs_free_energy


def calculate_ddg(gfe1, gfe2):
    """Calculate ΔΔG from two Gibbs free energies."""
    return gfe2 - gfe1


def process_ddg_optimized(sequences, pdb_files):
    """Compute ΔΔG for sequences using improved MD pipeline."""
    gibbs_free_energies = {}
    
    for i, pdb_file in enumerate(pdb_files):
        seq = sequences[i]
        print(f"\nProcessing sequence: {seq[:10]}...")
        
        modeller, system = load_pdb_structure(pdb_file)
        simulation, energy_samples, potential_energy = run_simulation(
            modeller, system,
            temperature=310*unit.kelvin,
            equil_steps=5000,
            md_steps=100000,
            sample_interval=1000
        )
        
        gibbs_free_energy = calculate_gibbs_free_energy(energy_samples, temperature=310*unit.kelvin)
        gibbs_free_energies[seq] = gibbs_free_energy
        
        print(f"Gibbs Free Energy for {seq[:10]}: {gibbs_free_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    # Calculate ΔΔG pairs
    ddg_results = {}
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1, seq2 = sequences[i], sequences[j]
            ddg = calculate_ddg(gibbs_free_energies[seq1], gibbs_free_energies[seq2])
            ddg_results[(seq1[:10], seq2[:10])] = ddg
            print(f"ΔΔG ({seq1[:10]} vs {seq2[:10]}): {ddg.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    return ddg_results


# Example usage
if __name__ == '__main__':
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAAHHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
    ]

    pdb_files = [
        'p_prepared1.pdb',
        'p_prepared2.pdb',
        'p_prepared3.pdb'
    ]

    ddg_results = process_ddg_optimized(sequences, pdb_files)

    print("\nFinal ΔΔG Results:")
    for seq_pair, ddg in ddg_results.items():
        print(f"ΔΔG for {seq_pair[0]} vs {seq_pair[1]}: {ddg.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
