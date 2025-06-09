# heuristic_ddg_pipeline.py. Under review. Not meant to be a rigorous calculation replacing FEP. 
import os
import numpy as np
import json
import openmm
from openmm import app, unit
from openmm.unit import kelvin, kilojoule_per_mole, picoseconds
from datetime import datetime

np.random.seed(42)

# ===== Utility Functions =====

def get_kB():
    """Returns Boltzmann constant in kJ/mol/K"""
    return 0.0083145 * kilojoule_per_mole / kelvin

def calculate_entropy_from_fluctuations(energy_values, temperature):
    """Estimate entropy using crude fluctuation approximation."""
    kB = get_kB()
    energies = np.array([e.value_in_unit(kilojoule_per_mole) for e in energy_values])
    fluct = np.var(energies)
    T = temperature.value_in_unit(kelvin)
    C_v = fluct / (kB * T**2)
    S = C_v * np.log(T)
    return S * kilojoule_per_mole / kelvin

def setup_logger(logfile="results/logs/run.log"):
    """Simple file logger."""
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    def log(msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(logfile, 'a') as f:
            f.write(f"[{timestamp}] {msg}\n")
        print(msg)
    return log

# ===== Core Simulation Functions =====

def load_structure(pdb_file, padding=1.0):
    """Loads and solvates a PDB structure."""
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB not found: {pdb_file}")
    
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber14/protein.ff14SB.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', padding=padding * unit.nanometer)
    
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds
    )
    return modeller, system

def run_md(modeller, system, temperature, equil_steps, md_steps, interval, save_dcd=False, dcd_filename="trajectory.dcd"):
    """Run MD simulation and return energy samples."""
    integrator = openmm.LangevinIntegrator(temperature, 1/picoseconds, 0.002*picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    log("Starting equilibration...")
    simulation.step(equil_steps)
    log("Equilibration complete. Starting production MD...")

    energy_samples = []

    if save_dcd:
        simulation.reporters.append(app.DCDReporter(dcd_filename, interval))

    for step in range(0, md_steps, interval):
        simulation.step(interval)
        state = simulation.context.getState(getEnergy=True)
        energy_samples.append(state.getPotentialEnergy())

    return energy_samples

def approximate_relative_free_energy(energies, temperature):
    """Estimate Helmholtz free energy using mean E and S."""
    e_vals = np.array([e.value_in_unit(kilojoule_per_mole) for e in energies])
    avg_e = np.mean(e_vals) * kilojoule_per_mole
    S = calculate_entropy_from_fluctuations(energies, temperature)
    F = avg_e - temperature * S
    return F

def calculate_ddg(free_energy_dict, sequences):
    """Calculate ΔΔG between sequence pairs."""
    ddgs = {}
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            seq1, seq2 = sequences[i], sequences[j]
            ddg = free_energy_dict[seq2] - free_energy_dict[seq1]
            ddgs[(seq1[:10], seq2[:10])] = ddg
    return ddgs

# ===== Runner =====

def process_sequences(sequences, pdb_paths, config):
    """Main ΔΔG processing logic."""
    T = config["temperature"] * kelvin
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    free_energies = {}

    for i, seq in enumerate(sequences):
        log(f"\nProcessing sequence: {seq[:10]}...")
        modeller, system = load_structure(pdb_paths[i])
        dcd_out = os.path.join(results_dir, f"{seq[:10]}.dcd") if config.get("save_dcd", False) else None
        energies = run_md(
            modeller,
            system,
            temperature=T,
            equil_steps=config["equil_steps"],
            md_steps=config["md_steps"],
            interval=config["sample_interval"],
            save_dcd=config.get("save_dcd", False),
            dcd_filename=dcd_out
        )
        F = approximate_relative_free_energy(energies, T)
        free_energies[seq] = F.value_in_unit(kilojoule_per_mole)
        log(f"Estimated F for {seq[:10]}: {free_energies[seq]:.2f} kJ/mol")

    ddgs = calculate_ddg(free_energies, sequences)

    with open(os.path.join(results_dir, "ddg_results.json"), 'w') as f:
        json.dump({f"{k[0]} vs {k[1]}": v for k, v in ddgs.items()}, f, indent=2)

    log("\nΔΔG Results:")
    for pair, ddg in ddgs.items():
        log(f"{pair[0]} vs {pair[1]}: ΔΔG ≈ {ddg:.2f} kJ/mol")

    return ddgs


# ===== Main Entrypoint =====

def log(msg):
    global logger
    logger(msg)

if __name__ == "__main__":
    logger = setup_logger()

    # Example input
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAAHHIGRGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK"
    ]

    pdb_files = [
        "p_prepared1.pdb",
        "p_prepared2.pdb",
        "p_prepared3.pdb"
    ]

    config = {
        "temperature": 310,         # in K
        "equil_steps": 5000,
        "md_steps": 50000,
        "sample_interval": 500,
        "save_dcd": True
    }

    ddg_results = process_sequences(sequences, pdb_files, config)
