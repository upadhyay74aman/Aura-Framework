import torch


class AuraMetricsEvaluator:
    """Calculates the optimization trade-offs between vector privacy defenses

    and semantic retrieval accuracy.
    """

    @staticmethod
    def calculate_search_utility(
        query_vector: torch.Tensor,
        document_database: torch.Tensor,
        defended_query_vector: torch.Tensor,
        top_k: int = 3,
    ) -> float:
        """Measures the alignment overlap between original and defended search indices.

        Returns a percentage value (1.0 = 100% search accuracy retained).
        """
        # Explicitly set dim=0 because we are comparing single 1D vectors instead of 2D batches
        similarity_metric = lambda x1, x2: torch.nn.functional.cosine_similarity(
            x1, x2, dim=0
        )

        # Ground Truth: Top-K search indices using the clean query
        orig_similarities = torch.stack(
            [similarity_metric(query_vector, doc) for doc in document_database]
        )
        base_top_k_indices = set(
            torch.topk(orig_similarities, k=top_k).indices.tolist()
        )

        # Defended Matrix: Top-K search indices using the protected query
        defended_similarities = torch.stack(
            [
                similarity_metric(defended_query_vector, doc)
                for doc in document_database
            ]
        )
        defended_top_k_indices = set(
            torch.topk(defended_similarities, k=top_k).indices.tolist()
        )

        # Calculate the Jaccard intersection hit-rate
        matches = base_top_k_indices.intersection(defended_top_k_indices)
        utility_retention = len(matches) / top_k
        return utility_retention

    @staticmethod
    def evaluate_vulnerability_profile(
        base_similarity: float, final_similarity: float
    ) -> dict:
        """Determines explicit threat categorization tiers based on data leakage profiles."""
        risk_tier = "SECURE"
        status_label = "SAFE"

        if final_similarity >= 0.85:
            risk_tier = "CRITICAL"
            status_label = "CRITICAL LEAK"
        elif final_similarity >= 0.50:
            risk_tier = "HIGH"
            status_label = "HIGH RISK"
        elif final_similarity >= 0.25:
            risk_tier = "MODERATE"
            status_label = "MODERATE MITIGATION"

        return {
            "max_proximity": f"{final_similarity * 100:.2f}% Cosine Similarity",
            "mitigation_delta": f"{(base_similarity - final_similarity) * 100:.2f}% Shaved",
            "tier": risk_tier,
            "status": status_label,
        }


# --- Standalone Logic Validation Check ---
if __name__ == "__main__":
    print("[METRICS ENGINE] Executing integration simulation test...")

    # Mock database setup: 5 documents, 64-dimensions each
    torch.manual_seed(42)
    mock_db = torch.nn.functional.normalize(torch.randn(5, 64), p=2, dim=-1)

    # Clean incoming query vector
    clean_query = mock_db[0] + torch.randn(64) * 0.1
    clean_query = torch.nn.functional.normalize(clean_query, p=2, dim=-1)

    # Defended incoming query (simulating added noise perturbation)
    defended_query = clean_query + torch.randn(64) * 0.2
    defended_query = torch.nn.functional.normalize(defended_query, p=2, dim=-1)

    retention_score = AuraMetricsEvaluator.calculate_search_utility(
        query_vector=clean_query,
        document_database=mock_db,
        defended_query_vector=defended_query,
        top_k=2,
    )

    print(f"Search Utility Retention Analysis: {retention_score * 100:.1f}% Match")