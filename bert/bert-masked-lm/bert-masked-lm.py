import numpy as np
from typing import Tuple


def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply BERT-style MLM masking.

    Selected positions:
    - 80% replaced with [MASK]
    - 10% replaced with a random token
    - 10% left unchanged

    Returns:
        masked_ids: input IDs after masking
        labels: original token IDs at selected positions,
                and -100 elsewhere
    """
    masked_ids = token_ids.copy()

    # -100 means "ignore this position when computing loss"
    labels = np.full_like(token_ids, -100)

    # Save original token IDs only for selected MLM positions
    labels[mask_positions] = token_ids[mask_positions]

    # 80%: replace selected tokens with [MASK]
    use_mask = mask_positions & (replace_probs < 0.8)

    # 10%: replace selected tokens with random tokens
    use_random = (
        mask_positions
        & (replace_probs >= 0.8)
        & (replace_probs < 0.9)
    )

    masked_ids[use_mask] = mask_token_id
    masked_ids[use_random] = random_tokens[use_random]

    # Values >= 0.9 remain unchanged
    return masked_ids, labels


class MLMHead:
    """Masked LM prediction head."""

    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict vocabulary logits.

        hidden_states shape:
            (batch, seq_len, hidden_size)

        output shape:
            (batch, seq_len, vocab_size)
        """
        return hidden_states @ self.W + self.b