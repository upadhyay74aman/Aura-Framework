import sys
import torch
import torch.nn as nn
import numpy as np
from core.defense import AuraDefenseSuite
from core.inversion import load_audit_vocabulary, run_inversion_attack
from core.metrics import AuraMetricsEvaluator
from utils.display import AuraConsoleUI
from rich.console import Console
from rich.table import Table

console = Console()


def generate_ascii_frontier_graph(steps: list, recovery_curve: list, utility_curve: list):
    """Renders a high-contrast text-based scatter plot tracking the safety elbow point."""
    print("\n" + "=" * 80)
    print("📈 PRIVACY-UTILITY FRONTIER CURVE (ASCII GRAPH)")
    print("   [R] = Token Recovery Rate (%)  │  [U] = RAG Search Utility (%)")
    print("=" * 80)
    
    # Render from 100% down to 0% in increments of 10
    for y_level in range(100, -1, -10):
        row_str = f"{y_level:3d}% │ "
        for idx, sigma in enumerate(steps):
            rec_val = recovery_curve[idx]
            util_val = utility_curve[idx]
            
            # Pick display glyphs based on proximity to the chart row coordinate
            has_r = abs(rec_val - y_level) < 5
            has_u = abs(util_val - y_level) < 5
            
            if has_r and has_u:
                row_str += " ❌  "  # Overlap collision point
            elif has_r:
                row_str += " [R] "
            elif has_u:
                row_str += " [U] "
            else:
                row_str += "  ·  "
        print(row_str)
        
    print("     └" + "─────" * len(steps))
    sigma_axis = "       " + " ".join([f"{s:.2f} " for s in steps])
    print(sigma_axis + " (Noise Sigma)")


def execute_full_framework_audit():
    AuraConsoleUI.print_banner()
    vocab_path = "data/vocabulary.txt"
    vocab_list = load_audit_vocabulary(vocab_path)
    EMBEDDING_DIM = 128

    print("[AURA SETUP] Ready for comprehensive privacy analysis.")
    user_input = input("📝 Enter a 3 or 4 word secret string to test: ")
    secret_phrase = user_input.lower().strip().split()

    if not secret_phrase:
        print("[ERROR] Input sequence cannot be empty.")
        sys.exit(1)

    # Dynamic vocabulary synchronization
    vocab_modified = False
    for word in secret_phrase:
        if word not in vocab_list:
            vocab_list.append(word)
            vocab_modified = True
    if vocab_modified:
        vocab_list = sorted(list(set(vocab_list)))
        with open(vocab_path, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab_list) + "\n")
    vocab_size = len(vocab_list)

    # 1. Establish True Victim Codebook Weights
    torch.manual_seed(1337)
    raw_victim_weights = torch.randn((vocab_size, EMBEDDING_DIM))
    victim_weights = nn.functional.normalize(raw_victim_weights, p=2, dim=-1)
    mock_database = victim_weights.clone()
    clean_target_vector = victim_weights[[vocab_list.index(w) for w in secret_phrase]]

    # 2. Establish Attacker's Black-Box Surrogate Space
    surrogate_deviation = torch.randn_like(victim_weights) * 0.03
    attacker_surrogate_weights = nn.functional.normalize(
        victim_weights + surrogate_deviation, p=2, dim=-1
    )

    # 3. Primary Audit Phase (Detailed Trace Logging)
    print("\n[AUDIT] Launching Baseline Black-Box Attacker Optimization Run...")
    baseline_result = run_inversion_attack(
        target_vector=clean_target_vector,
        vocab_path=vocab_path,
        embedding_matrix=attacker_surrogate_weights,
        clean_reference=clean_target_vector,
        verbose=True,
    )

    print("\n[AUDIT] Launching Defended Run (Noise Sigma = 0.15)...")
    defended_target = AuraDefenseSuite.inject_gaussian_noise(clean_target_vector, sigma=0.15)
    defended_result = run_inversion_attack(
        target_vector=defended_target,
        vocab_path=vocab_path,
        embedding_matrix=attacker_surrogate_weights,
        clean_reference=clean_target_vector,
        verbose=True,
    )

    # 4. Automated Noise Sweep Across Monte Carlo Randomized Trials
    print("\n" + "=" * 80)
    print("📊 RUNNING MULTI-METRIC NOISE SWEEP FRONTIER")
    print("=" * 80)

    noise_steps = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

    sweep_table = Table(title="\nPRIVACY VS UTILITY DEGRADATION MATRIX")
    sweep_table.add_column("Noise Sigma", style="cyan", justify="center")
    sweep_table.add_column("Raw Embedding Similarity\n(Published vs Clean)", style="blue", justify="center")
    sweep_table.add_column("Token Recovery Rate\n(Mean ± Std)", style="green", justify="center")
    sweep_table.add_column("Discrete Token Similarity\n(Decoded vs Clean)", style="yellow", justify="center")
    sweep_table.add_column("RAG Utility Retention\n(Search Match %)", style="magenta", justify="center")

    graph_recovery = []
    graph_utility = []

    for sigma in noise_steps:
        trial_recovery_rates = []
        trial_discrete_sims = []
        trial_utilities = []
        trial_raw_embedding_sims = []

        for seed_offset in range(10):
            torch.manual_seed(42 + seed_offset)

            # Re-generate defense matrix configuration
            def_vec = AuraDefenseSuite.inject_gaussian_noise(clean_target_vector, sigma=sigma)

            # Metric Upgrade: Calculate True Raw Embedding Cosine Distance BEFORE decryption
            raw_emb_sim = torch.nn.functional.cosine_similarity(def_vec, clean_target_vector, dim=-1).mean().item()
            trial_raw_embedding_sims.append(raw_emb_sim)

            # Silently run target inversion attack
            res = run_inversion_attack(
                target_vector=def_vec,
                vocab_path=vocab_path,
                embedding_matrix=attacker_surrogate_weights,
                verbose=False,
            )

            # Calculate Exact Match Token Recovery Rate
            matches = sum(1 for true, pred in zip(secret_phrase, res["words"]) if true == pred)
            trial_recovery_rates.append(matches / len(secret_phrase))

            # Calculate Discrete Coordinate Structural Similarity
            res_idcs = [vocab_list.index(w) for w in res["words"]]
            inferred_vecs = victim_weights[res_idcs]
            sim = torch.nn.functional.cosine_similarity(inferred_vecs, clean_target_vector, dim=-1).mean().item()
            trial_discrete_sims.append(sim)

            # Calculate Local Search Application Performance Overlap
            util = AuraMetricsEvaluator.calculate_search_utility(
                query_vector=clean_target_vector[0],
                document_database=mock_database,
                defended_query_vector=def_vec[0],
                top_k=3,
            )
            util_score = 1.0 if util >= 1.0 else util  # normalize bounds
            trial_utilities.append(util_score)

        # Parse metrics array profiles
        mean_raw_emb = np.mean(trial_raw_embedding_sims) * 100
        mean_rec = np.mean(trial_recovery_rates) * 100
        std_rec = np.std(trial_recovery_rates) * 100
        mean_sim = np.mean(trial_discrete_sims) * 100
        mean_util = np.mean(trial_utilities) * 100

        graph_recovery.append(mean_rec)
        graph_utility.append(mean_util)

        sweep_table.add_row(
            f"{sigma:.2f}",
            f"{mean_raw_emb:.1f}% Cos Sim",
            f"{mean_rec:.1f}% ± {std_rec:.1f}%",
            f"{mean_sim:.1f}% Cos Sim",
            f"{mean_util:.1f}% Match",
        )

    console.print(sweep_table)
    
    # 5. Render the ASCII Graph Visualizer
    generate_ascii_frontier_graph(noise_steps, graph_recovery, graph_utility)
    print("\n[+] Audit Framework Loop Successfully Concluded.\n")


if __name__ == "__main__":
    execute_full_framework_audit()