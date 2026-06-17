import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rich.console import Console
from rich.table import Table
from core.defense import AuraDefenseSuite
from core.inversion import (
    load_audit_vocabulary, 
    run_inversion_attack, 
    run_transformer_inversion_attack
)
from core.metrics import AuraMetricsEvaluator
from utils.display import AuraConsoleUI

console = Console()

def generate_ascii_frontier_graph(steps: list, recovery_curve: list, utility_curve: list):
    # [Keep your existing generate_ascii_frontier_graph function here exactly as it was]
    print("\n" + "=" * 80)
    print("📈 PRIVACY-UTILITY FRONTIER CURVE (ASCII GRAPH)")
    print("   [R] = Token Recovery Rate (%)  │  [U] = RAG Search Utility (%)")
    print("=" * 80)
    for y_level in range(100, -1, -10):
        row_str = f"{y_level:3d}% │ "
        for idx, sigma in enumerate(steps):
            rec_val = recovery_curve[idx]
            util_val = utility_curve[idx]
            has_r = abs(rec_val - y_level) < 5
            has_u = abs(util_val - y_level) < 5
            if has_r and has_u: row_str += " ❌  "
            elif has_r: row_str += " [R] "
            elif has_u: row_str += " [U] "
            else: row_str += "  ·  "
        print(row_str)
    print("     └" + "─────" * len(steps))
    print("       " + " ".join([f"{s:.2f} " for s in steps]) + " (Noise Sigma)")


def execute_sandbox_sweep():
    """Mode 1: The fast linear simulation and noise metric sweep."""
    AuraConsoleUI.print_banner()
    vocab_path = "data/vocabulary.txt"
    vocab_list = load_audit_vocabulary(vocab_path)
    EMBEDDING_DIM = 128

    print("[AURA SETUP] Ready for sandbox privacy sweep.")
    user_input = input("📝 Enter a 3 or 4 word secret string to test: ")
    secret_phrase = user_input.lower().strip().split()
    if not secret_phrase: sys.exit(1)

    for word in secret_phrase:
        if word not in vocab_list: vocab_list.append(word)
    vocab_list = sorted(list(set(vocab_list)))
    with open(vocab_path, "w", encoding="utf-8") as f: f.write("\n".join(vocab_list) + "\n")
    
    torch.manual_seed(1337)
    raw_victim_weights = torch.randn((len(vocab_list), EMBEDDING_DIM))
    victim_weights = nn.functional.normalize(raw_victim_weights, p=2, dim=-1)
    clean_target_vector = victim_weights[[vocab_list.index(w) for w in secret_phrase]]
    
    surrogate_deviation = torch.randn_like(victim_weights) * 0.03
    attacker_surrogate_weights = nn.functional.normalize(victim_weights + surrogate_deviation, p=2, dim=-1)

    print("\n[AUDIT] Launching Baseline Run...")
    run_inversion_attack(clean_target_vector, vocab_path, attacker_surrogate_weights, clean_reference=clean_target_vector, verbose=True)

    print("\n" + "=" * 80)
    print("📊 RUNNING MULTI-METRIC NOISE SWEEP FRONTIER")
    print("=" * 80)

    noise_steps = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
    graph_recovery, graph_utility = [], []

    for sigma in noise_steps:
        trial_recovery_rates, trial_utilities = [], []
        for seed_offset in range(10):
            torch.manual_seed(42 + seed_offset)
            def_vec = AuraDefenseSuite.inject_gaussian_noise(clean_target_vector, sigma=sigma)
            res = run_inversion_attack(def_vec, vocab_path, attacker_surrogate_weights, verbose=False)
            
            matches = sum(1 for true, pred in zip(secret_phrase, res["words"]) if true == pred)
            trial_recovery_rates.append(matches / len(secret_phrase))
            trial_utilities.append(AuraMetricsEvaluator.calculate_search_utility(clean_target_vector[0], victim_weights.clone(), def_vec[0], 3))

        graph_recovery.append(np.mean(trial_recovery_rates) * 100)
        graph_utility.append(np.mean(trial_utilities) * 100)

    generate_ascii_frontier_graph(noise_steps, graph_recovery, graph_utility)
    print("\n[+] Sandbox Audit Concluded.\n")


def execute_real_world_audit():
    """Mode 2: High-Fidelity Hugging Face Transformer Inversion."""
    AuraConsoleUI.print_banner()
    print("\n[!] WARNING: Initiating Real-World Transformer Target")
    print("[INFO] Target: sentence-transformers/all-MiniLM-L6-v2")
    print("[INFO] Mechanism: Gumbel-Softmax Forward-Pass Gradient Optimization")
    
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("\n[ERROR] Hugging Face 'transformers' library not found.")
        print("Run: pip install transformers")
        sys.exit(1)

    print("\n⏳ Downloading/Loading Model Weights (~80MB)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval() # Freeze dropout

    user_input = input("\n📝 Enter a short secret phrase to extract (e.g., 'root password admin'): ")
    if not user_input.strip(): sys.exit(1)

    # 1. Simulate the RAG System producing a pooled target vector
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        target_pooled_vector = torch.sum(token_embeddings * attention_mask, 1) / torch.clamp(attention_mask.sum(1), min=1e-9)
        target_pooled_vector = F.normalize(target_pooled_vector, p=2, dim=1)

    seq_len = inputs['input_ids'].shape[1] - 2  # Subtracting [CLS] and [SEP] tokens

    # 2. Launch the deep inversion attack
    print("\n[🔥 ATTACK] Pushing continuous gradients through discrete pooling layers...")
    recovered_words = run_transformer_inversion_attack(
        target_pooled_vector=target_pooled_vector,
        model=model,
        tokenizer=tokenizer,
        seq_len=seq_len,
        epochs=300,
        lr=0.5
    )

    print("\n" + "="*50)
    print("🎯 REAL-WORLD EXTRACTION RESULTS")
    print("="*50)
    print(f"True Plaintext : {user_input}")
    print(f"Extracted Text : {' '.join(recovered_words).replace('[CLS]', '').replace('[SEP]', '').strip()}")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AURA Vector Auditing Framework")
    parser.add_argument("--real-world-audit", action="store_true", help="Run deep Hugging Face transformer inversion")
    args = parser.parse_args()

    if args.real_world_audit:
        execute_real_world_audit()
    else:
        execute_sandbox_sweep()