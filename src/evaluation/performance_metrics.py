"""Model performance evaluation metrics."""
from typing import List, Dict, Any
import numpy as np


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Model loss value
        
    Returns:
        Perplexity score (lower is better)
    """
    return np.exp(loss)


def calculate_text_statistics(texts: List[str]) -> Dict[str, Any]:
    """
    Calculate text length statistics.
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with length statistics
    """
    if not texts:
        return {
            "count": 0,
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0
        }
    
    lengths = [len(text) for text in texts]
    
    return {
        "count": len(texts),
        "avg_length": round(sum(lengths) / len(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths)
    }
