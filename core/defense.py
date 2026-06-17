import torch


class AuraDefenseSuite:
    """Applies privacy-preserving mutations to high-dimensional embeddings

    to block gradient-guided token reconstruction loops.
    """

    @staticmethod
    def inject_gaussian_noise(
        vectors: torch.Tensor, sigma: float = 0.05
    ) -> torch.Tensor:
        """Adds calculated standard normal noise to coordinates.

        Blunts fine-grained gradient tracking while preserving overall semantic
        direction.
        """
        noise = torch.randn_like(vectors) * sigma
        perturbed_vectors = vectors + noise

        # Crucial: Re-normalize vectors back onto the hypersphere
        # Vector operations like cosine similarity depend on normalized scaling
        return torch.nn.functional.normalize(perturbed_vectors, p=2, dim=-1)

    @staticmethod
    def apply_quantization(vectors: torch.Tensor, bits: int = 4) -> torch.Tensor:
        """Reduces 32-bit floating-point coordinates into low-bit discrete integer steps.

        Destores the smooth gradient contours required by optimization decoders.
        """
        levels = (2**bits) - 1
        min_val, max_val = vectors.min(), vectors.max()

        if torch.equal(min_val, max_val):
            return vectors

        # Map continuous float variables into discrete bucket scales
        quantized = torch.round(
            (vectors - min_val) / (max_val - min_val) * levels
        )

        # De-quantize back to standard float format for model parsing compatibility
        dequantized = (quantized / levels) * (max_val - min_val) + min_val

        return torch.nn.functional.normalize(dequantized, p=2, dim=-1)


# --- Quick Independent Component Test Verification ---
if __name__ == "__main__":
    print("[DEFENSE MODULE] Running standalone transform checks...")
    test_tensor = torch.randn(1, 5)
    normalized_test = torch.nn.functional.normalize(test_tensor, p=2, dim=-1)

    print(f"Original Vector  : {normalized_test.tolist()}")

    # Test noise perturbation
    noisy_output = AuraDefenseSuite.inject_gaussian_noise(
        normalized_test, sigma=0.1
    )
    print(f"Perturbed (Noise): {noisy_output.tolist()}")

    # Test bit quantization
    quant_output = AuraDefenseSuite.apply_quantization(
        normalized_test, bits=4
    )
    print(f"Quantized (4-Bit): {quant_output.tolist()}")