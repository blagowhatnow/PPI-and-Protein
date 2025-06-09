#This code is experimental
#Might need further review 
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from Bio.Align import PairwiseAligner
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import substitution_matrices
import openmm
from openmm import app, unit


# ==== STRUCTURE LOADING ====

def load_pdb_structure(pdb_file):
    if not os.path.exists(pdb_file):
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    pdb = app.PDBFile(pdb_file)
    forcefield = app.ForceField('amber99sb.xml', 'amber99_obc.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff)
    return modeller, system


# ==== ENERGY CALCULATION ====

def calculate_distance(pos1, pos2):
    return np.linalg.norm(pos1 - pos2)


def extract_nonbonded_force(system):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    raise ValueError("NonbondedForce not found in the system.")


def calculate_atom_interaction(index1, index2, positions, nonbonded_force, cutoff=0.6):
    q1, sigma1, eps1 = [x.value_in_unit(y) for x, y in zip(nonbonded_force.getParticleParameters(index1), 
                                                            [unit.elementary_charge, unit.nanometer, unit.kilojoule_per_mole])]
    q2, sigma2, eps2 = [x.value_in_unit(y) for x, y in zip(nonbonded_force.getParticleParameters(index2), 
                                                            [unit.elementary_charge, unit.nanometer, unit.kilojoule_per_mole])]

    pos1, pos2 = positions[index1], positions[index2]
    r = calculate_distance(pos1, pos2)
    if r < cutoff:
        return 0, 0

    # Lennard-Jones
    epsilon = np.sqrt(eps1 * eps2)
    sigma = 0.5 * (sigma1 + sigma2)
    lj = 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

    # Coulombic energy
    k_e = 8.9875517923e9
    q1_coulomb = q1 * 1.602e-19
    q2_coulomb = q2 * 1.602e-19
    coulomb_joule = k_e * q1_coulomb * q2_coulomb / (r * 1e-9)
    coulomb_kjmol = (coulomb_joule / 1000) * 6.022e23

    return lj, coulomb_kjmol

def calculate_residue_interaction(modeller, system):
    integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    nb_force = extract_nonbonded_force(system)
    residues = list(modeller.topology.residues())
    positions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
    n = len(residues)
    interaction_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            energy = 0
            for atom1 in residues[i].atoms():
                for atom2 in residues[j].atoms():
                    lj, coul = calculate_atom_interaction(atom1.index, atom2.index, positions, nb_force)
                    energy += lj + coul
            interaction_matrix[i, j] = interaction_matrix[j, i] = energy

    return interaction_matrix


# ==== ALIGNMENT + SCORING ====

def align_sequences(sequences):
    aligner = PairwiseAligner()
    aligner.mode = 'global'

    # Start with the first sequence as the initial alignment base
    msa = [sequences[0]]

    for i in range(1, len(sequences)):
        aligned = aligner.align(msa[0], sequences[i])[0]

        aligned_seq1 = str(aligned[0])
        aligned_seq2 = str(aligned[1])

        # Update the reference alignment with gaps if needed
        updated_msa = []
        ref_index = 0
        new_index = 0

        while ref_index < len(aligned_seq1) and new_index < len(aligned_seq2):
            a1 = aligned_seq1[ref_index]
            a2 = aligned_seq2[new_index]

            for j in range(len(msa)):
                if len(updated_msa) <= j:
                    updated_msa.append("")

                if a1 == '-':
                    updated_msa[j] += '-'
                else:
                    updated_msa[j] += msa[j][0]
                    msa[j] = msa[j][1:]

            ref_index += 1
            new_index += 1

        msa = updated_msa + [aligned_seq2]

    # Convert to SeqRecord for compatibility
    msa_records = [SeqRecord(Seq(seq), id=f"seq_{i}") for i, seq in enumerate(msa)]

    # Alignment score matrix (optional here)
    matrix = np.zeros((len(msa_records), len(msa_records)))

    return msa_records, matrix

def build_substitution_matrix():
    raw = substitution_matrices.load('BLOSUM62')
    matrix = {}
    for (a1, a2), score in raw.items():
        matrix[(a1, a2)] = matrix[(a2, a1)] = score
    return matrix


def calculate_hybrid_energy(seq1, seq2, mat1, mat2, substitution_matrix=None, alpha=1.0, beta=1.0):
    """
    Calculate hybrid energy combining evolutionary and physical interaction energies.
    
    Parameters:
    - seq1, seq2: aligned sequences (strings, equal length)
    - mat1, mat2: physical interaction matrices (numpy arrays)
    - substitution_matrix: dict with substitution scores (e.g., BLOSUM62)
    - alpha, beta: weights for evolutionary and physical energies
    
    Returns:
    - total_energy: combined weighted energy (float)
    - evo_energy_avg: normalized average evolutionary score per residue pair
    - phys_energy_avg: normalized average physical energy per residue pair
    """
    seq1, seq2 = str(seq1), str(seq2)
    assert len(seq1) == len(seq2), "Aligned sequences must be equal length."
    L = len(seq1)

    evo_energy_sum = 0.0
    phys_energy_sum = 0.0
    valid_pairs = 0  # count residue pairs without gaps

    # Calculate sum of evolutionary and physical energies over residue pairs (i<j)
    for i in range(L):
        for j in range(i + 1, L):
            # skip pairs with gaps
            if '-' in (seq1[i], seq2[i], seq1[j], seq2[j]):
                continue
            
            valid_pairs += 1
            
            # Evolutionary energy: average substitution scores for the two positions
            if substitution_matrix:
                evo_score_1 = substitution_matrix.get((seq1[i], seq2[i]), substitution_matrix.get((seq2[i], seq1[i]), 0))
                evo_score_2 = substitution_matrix.get((seq1[j], seq2[j]), substitution_matrix.get((seq2[j], seq1[j]), 0))
                evo_energy_sum += 0.5 * (evo_score_1 + evo_score_2)
            else:
                # fallback: 1 if residues differ, 0 if same
                evo_energy_sum += 1 if (seq1[i] != seq2[i] and seq1[j] != seq2[j]) else 0
            
            # Physical energy: sum interaction energies from matrices
            try:
                phys_energy_sum += mat1[i, j] + mat2[i, j]
            except IndexError:
                # If interaction matrices are smaller (due to gaps), skip or treat as zero
                phys_energy_sum += 0

    if valid_pairs == 0:
        raise ValueError("No valid residue pairs without gaps to calculate energy.")

    # Normalize energies per residue pair
    evo_energy_avg = evo_energy_sum / valid_pairs
    phys_energy_avg = phys_energy_sum / valid_pairs

    # OPTIONAL: Normalize physical energy scale for typical magnitude (example: divide by 10)
    phys_energy_avg /= 10.0

    # Combine weighted energies
    # Since evolutionary substitution scores are often positive for similar pairs,
    # and physical energy tends to be negative for favorable interactions,
    # we invert evo energy to treat it as a "cost" (lower is better)
    total_energy = alpha * (-evo_energy_avg) + beta * phys_energy_avg

    return total_energy



# ==== MAIN WORKFLOW ====

def calculate_msa_fitness(msa, pdb_files, alignment_matrix, alpha=1.0, beta=1.0, max_workers=4):
    fitness = {}
    interaction_matrices = {}
    for pdb in pdb_files:
        if pdb not in interaction_matrices:
            modeller, system = load_pdb_structure(pdb)
            interaction_matrices[pdb] = calculate_residue_interaction(modeller, system)

    substitution_matrix = build_substitution_matrix()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i in range(len(msa)):
            for j in range(i + 1, len(msa)):
                seq1, seq2 = msa[i].seq, msa[j].seq
                m1, m2 = interaction_matrices[pdb_files[i]], interaction_matrices[pdb_files[j]]
                futures[(i, j)] = executor.submit(
                    calculate_hybrid_energy, seq1, seq2, m1, m2, substitution_matrix, alpha, beta
                )

        for (i, j), fut in futures.items():
            try:
                fitness[(i, j)] = fut.result(timeout=600)
            except Exception as e:
                print(f"Error for pair ({i}, {j}): {e}")

    return fitness


def process_sequences_and_pdbs(sequences, pdb_files, alpha=1.0, beta=1.0):
    msa, align_scores = align_sequences(sequences)
    fitness = calculate_msa_fitness(msa, pdb_files, align_scores, alpha, beta)
    print("Hybrid Fitness Energies:")
    for (i, j), energy in fitness.items():
        print(f"Pair ({i}, {j}) → Energy: {energy:.3f}")


# ==== USAGE EXAMPLE ====

if __name__ == "__main__":
    sequences = [
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKIGKKGGGELASK",
        "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFIGGIGLASK" 
    ]
    pdb_files = ["m1.pdb", "m2.pdb", "m3.pdb"]
    process_sequences_and_pdbs(sequences, pdb_files)
