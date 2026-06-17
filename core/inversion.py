import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.display import AuraConsoleUI


def load_audit_vocabulary(vocab_path: str) -> list:
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
    """Fast, linear continuous-relaxation attack for the sandbox baseline sweep."""
    vocab_list = load_audit_vocabulary(vocab_path)
    num_tokens = target_vector.size(0)
    vocab_size = len(vocab_list)

    soft_tokens = nn.Parameter(torch.randn(num_tokens, vocab_size) * 0.01)
    optimizer = torch.optim.Adam([soft_tokens], lr=learning_rate)

    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = torch.softmax(soft_tokens, dim=-1)
        predicted_vecs = torch.matmul(probs, embedding_matrix)
        similarity = F.cosine_similarity(predicted_vecs, target_vector, dim=-1)
        loss = (1.0 - similarity).mean()
        loss.backward()
        optimizer.step()

        if verbose and epoch % 25 == 0:
            current_indices = torch.argmax(soft_tokens, dim=-1).tolist()
            current_words = [vocab_list[idx] for idx in current_indices]
            display_sim = (
                F.cosine_similarity(predicted_vecs, clean_reference, dim=-1).mean().item()
                if clean_reference is not None else similarity.mean().item()
            )
            AuraConsoleUI.render_matrix_row(epoch, loss.item(), display_sim, current_words)

    final_probs = torch.softmax(soft_tokens, dim=-1)
    final_indices = torch.argmax(final_probs, dim=-1).tolist()
    top_words = [vocab_list[idx] for idx in final_indices]
    
    top_k_data = []
    for pos in range(num_tokens):
        pos_vals, pos_idcs = torch.topk(final_probs[pos], k=min(4, vocab_size))
        pos_distribution = [(vocab_list[idx], val) for val, idx in zip(pos_vals.tolist(), pos_idcs.tolist())]
        top_k_data.append(pos_distribution)

    return {"words": top_words, "top_k": top_k_data}


def run_transformer_inversion_attack(
    target_pooled_vector: torch.Tensor,
    model: nn.Module,
    tokenizer,
    seq_len: int,
    epochs: int = 300,
    lr: float = 0.5,
    initial_tau: float = 1.2,
    min_tau: float = 0.05
) -> list:
    """
    High-fidelity Gray-Box attack against a real Transformer's mean-pooled outputs.
    Implements Exponential Temperature Annealing to break out of local minima.
    """
    vocab_size = model.config.vocab_size
    word_embeddings = model.get_input_embeddings().weight.detach()

    # Initialize logits with a slightly higher variance to encourage exploration
    logits = nn.Parameter(torch.randn(1, seq_len, vocab_size) * 0.05)
    optimizer = torch.optim.Adam([logits], lr=lr)
    
    # Simple step decay for learning rate to stabilize late-stage optimization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long, device=logits.device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Exponentially decay temperature (Annealing Schedule)
        tau = max(initial_tau * (min_tau / initial_tau) ** (epoch / epochs), min_tau)

        # Gumbel-Softmax relaxation with dynamic temperature
        one_hot = F.gumbel_softmax(logits, tau=tau, hard=True)

        # Reconstruct continuous embeddings from the one-hot selection
        inputs_embeds = torch.matmul(one_hot, word_embeddings)

        # Forward pass through the transformer layers
        outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        # Execute mean pooling calculation
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled_output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = F.normalize(pooled_output, p=2, dim=1)

        # Minimize cosine distance
        similarity = F.cosine_similarity(pooled_output, target_pooled_vector, dim=-1)
        loss = 1.0 - similarity.mean()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 50 == 0:
            current_ids = torch.argmax(logits, dim=-1)[0].tolist()
            current_guess = tokenizer.decode(current_ids)
            print(f" [Step {epoch:03d}] Loss: {loss.item():.4f} │ Sim: {similarity.item():.4f} │ Temp (Tau): {tau:.3f} │ Guess: '{current_guess}'")

    final_ids = torch.argmax(logits, dim=-1)[0].tolist()
    return tokenizer.decode(final_ids).split()