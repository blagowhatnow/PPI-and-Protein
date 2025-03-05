#This code is experimental
#Might need further review


import openmm
from openmm import app, unit
import numpy as np
import os

# Load the PDB structure for a heterodimer with or without solvent
def load_pdb_structure(pdb_file, solvent=True):
    """Load a PDB structure file and set up the system with or without solvent."""
    if not os.path.exists(pdb_file):
        raise ValueError(f"PDB file not found: {pdb_file}")
    
    pdb = app.PDBFile(pdb_file)
    
    # Force field parameters: Protein + Solvent (if solvent=True) or Protein only (if solvent=False)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml') if solvent else app.ForceField('amber14/protein.ff14SB.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    
    # If solvent is required, add water model
    if solvent:
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * unit.nanometer)  # Solvate with water
    
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    return modeller, system

# Run energy minimization
def run_minimization(modeller, system):
    """Minimize the energy of the system."""
    integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)  # Integrator for MD simulation
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Minimize the energy
    print("Minimizing system energy...")
    simulation.minimizeEnergy()
    
    # Obtain the final energy of the minimized system
    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy()
    
    return final_energy

# Run a full MD simulation
def run_md_simulation(modeller, system, temperature=300*unit.kelvin, steps=10000):
    """Run a full MD simulation and return the final potential energy."""
    integrator = openmm.LangevinIntegrator(temperature, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Set up reporters for energy collection during the simulation
    energy_reporter = app.StateDataReporter('output.log', 1000, step=True, potentialEnergy=True, temperature=True)
    simulation.reporters.append(energy_reporter)
    
    # Run the simulation for a defined number of steps
    print(f"Running MD simulation for {steps} steps...")
    simulation.step(steps)
    
    # Obtain the final state and energy of the simulation
    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy()
    
    return final_energy

def run_equilibration(modeller, system, temperature=300*unit.kelvin, equilibration_steps=5000):
    """Run an equilibration phase to relax the system before the production run."""
    integrator = openmm.LangevinIntegrator(temperature, 1 / unit.picosecond, 0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Apply position restraints to the solute during equilibration
    force = openmm.CustomExternalForce('0.5*k*(x^2 + y^2 + z^2)')
    force.addPerParticleParameter('k')
    for atom in modeller.topology.atoms():
        force.addParticle(atom.index, [1.0])  # Apply restraint with force constant k=1.0 (in kcal/mol/nm^2)
    system.addForce(force)
    
    # Run equilibration for the specified number of steps
    print(f"Running equilibration for {equilibration_steps} steps...")
    simulation.step(equilibration_steps)
    
    # Remove position restraints after equilibration
    system.removeForce(system.getNumForces()-1)
    
    # Return the final equilibrated system energy
    state = simulation.context.getState(getEnergy=True)
    final_energy = state.getPotentialEnergy()
    
    return final_energy

# Function to calculate solvent reorganization energy from MD simulations
def calculate_solvent_reorganization_energy(pdb_file, steps=200000, equilibration_steps=9000):
    """Calculate the solvent reorganization energy using MD simulation."""
    
    # Step 1: Load and minimize the system in vacuum (no solvent)
    print("Running minimization for vacuum state (no solvent)...")
    modeller_vacuum, system_vacuum = load_pdb_structure(pdb_file, solvent=False)
    minimization_vacuum_energy = run_minimization(modeller_vacuum, system_vacuum)
    print(f"Energy in vacuum after minimization: {minimization_vacuum_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    # Step 2: Solvate the system after minimization
    print("\nSolvating the system...")
    modeller_solvent, system_solvent = load_pdb_structure(pdb_file, solvent=True)
    
    # Step 3: Equilibrate the solvated system before production
    print("\nEquilibrating the system with solvent...")
    equilibrated_energy = run_equilibration(modeller_solvent, system_solvent, equilibration_steps=equilibration_steps)
    print(f"Energy after equilibration: {equilibrated_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    # Step 4: Run MD simulation in vacuum (no solvent)
    print("\nRunning MD simulation in vacuum state (no solvent)...")
    md_vacuum_energy = run_md_simulation(modeller_vacuum, system_vacuum, steps=steps)
    print(f"Energy in vacuum after MD simulation: {md_vacuum_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    # Step 5: Run MD simulation with solvent (water)
    print("\nRunning MD simulation in solvent state (with water)...")
    md_solvent_energy = run_md_simulation(modeller_solvent, system_solvent, steps=steps)
    print(f"Energy in solvent after MD simulation: {md_solvent_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    # Step 6: Calculate the solvent reorganization energy
    reorganization_energy = md_solvent_energy - md_vacuum_energy
    print(f"\nSolvent reorganization energy: {reorganization_energy.value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    
    return reorganization_energy

# Example usage:
if __name__ == "__main__":
    pdb_file = "p_prepared2.pdb"  # Replace with your actual PDB file for the heterodimer
    reorganization_energy = calculate_solvent_reorganization_energy(pdb_file, steps=10000, equilibration_steps=5000)