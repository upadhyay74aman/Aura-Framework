import torch
import torch.nn as nn
from utils.display import AuraConsoleUI


def load_audit_vocabulary(vocab_path: str) -> list:
    """Loads and returns structural string tokens from local disk."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_inversion_attack(
    target_vector: torch.Tensor,
    vocab_path: str,
    embedding_matrix: torch.Tensor,
    epochs: int = 101,
    learning_rate: float = 0.15,
    clean_reference: torch.Tensor = None,
    verbose: bool = True,
) -> dict:
    """Performs continuous relaxation optimization.

    Returns a telemetry dictionary containing argmax selections and complete top-K probability matrices.
    """
    vocab_list = load_audit_vocabulary(vocab_path)
    num_tokens = target_vector.size(0)
    vocab_size = len(vocab_list)

    # Continuous relaxation parameter tracking allocation
    soft_tokens = nn.Parameter(torch.randn(num_tokens, vocab_size) * 0.01)
    optimizer = torch.optim.Adam([soft_tokens], lr=learning_rate)
    similarity_metric = torch.nn.functional.cosine_similarity

    for epoch in range(epochs):
        optimizer.zero_grad()

        probs = torch.softmax(soft_tokens, dim=-1)
        predicted_vecs = torch.matmul(probs, embedding_matrix)

        similarity = similarity_metric(predicted_vecs, target_vector, dim=-1)
        loss = (1.0 - similarity).mean()

        loss.backward()
        optimizer.step()

        if verbose and epoch % 25 == 0:
            current_indices = torch.argmax(soft_tokens, dim=-1).tolist()
            current_words = [vocab_list[idx] for idx in current_indices]

            if clean_reference is not None:
                display_sim = (
                    similarity_metric(predicted_vecs, clean_reference, dim=-1)
                    .mean()
                    .item()
                )
            else:
                display_sim = similarity.mean().item()

            AuraConsoleUI.render_matrix_row(
                step=epoch,
                loss=loss.item(),
                similarity=display_sim,
                words=current_words,
            )

    final_probs = torch.softmax(soft_tokens, dim=-1)
    final_indices = torch.argmax(final_probs, dim=-1).tolist()
    top_words = [vocab_list[idx] for idx in final_indices]

    # Compute a Top-4 token probability readout for every word position
    top_k_data = []
    for pos in range(num_tokens):
        pos_vals, pos_idcs = torch.topk(final_probs[pos], k=min(4, vocab_size))
        pos_distribution = []
        for val, idx in zip(pos_vals.tolist(), pos_idcs.tolist()):
            pos_distribution.append((vocab_list[idx], val))
        top_k_data.append(pos_distribution)

    return {"words": top_words, "top_k": top_k_data}