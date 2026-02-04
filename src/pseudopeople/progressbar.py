"""
Progress bar utilities for pseudopeople.

Uses tqdm.auto which automatically selects the appropriate tqdm implementation
(tqdm.notebook for Jupyter, standard tqdm otherwise).
"""

from tqdm.auto import tqdm

__all__ = ["tqdm"]
