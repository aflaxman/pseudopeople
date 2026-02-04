"""
Progress bar utilities for pseudopeople.

Provides a unified interface for progress bars that automatically detects
whether code is running in a Jupyter notebook and uses the appropriate
tqdm implementation (tqdm.notebook for Jupyter, standard tqdm otherwise).
"""

from typing import Iterable, Optional


def _is_notebook() -> bool:
    """
    Detect whether code is running in a Jupyter notebook environment.

    Returns True if running in Jupyter notebook/lab, False otherwise.
    """
    try:
        # Check if IPython is available
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False

        # Check for ZMQInteractiveShell which indicates Jupyter
        shell_class = shell.__class__.__name__
        if shell_class == "ZMQInteractiveShell":
            return True
        elif shell_class == "TerminalInteractiveShell":
            return False
        else:
            # Unknown shell type, default to standard tqdm
            return False
    except (ImportError, NameError):
        return False


def _get_tqdm_class():
    """
    Get the appropriate tqdm class based on the runtime environment.

    Returns tqdm.notebook.tqdm if in Jupyter, otherwise tqdm.tqdm.
    """
    if _is_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    return tqdm


def progress_bar(
    iterable: Iterable,
    desc: Optional[str] = None,
    unit: str = "it",
    leave: bool = True,
    position: Optional[int] = None,
    total: Optional[int] = None,
    disable: bool = False,
):
    """
    Create a progress bar for an iterable.

    Automatically selects the appropriate tqdm implementation based on
    whether code is running in a Jupyter notebook.

    Parameters
    ----------
    iterable : Iterable
        The iterable to wrap with a progress bar.
    desc : str, optional
        Description text shown before the progress bar.
    unit : str, default "it"
        Unit name for the items being iterated.
    leave : bool, default True
        If True, keeps the progress bar visible after completion.
        Set to False for nested inner progress bars.
    position : int, optional
        Position of the progress bar (for nested bars).
        0 = outermost bar, 1 = first nested bar, etc.
        For terminal tqdm, this determines the line number.
        For Jupyter notebook, this is used but behavior may differ.
    total : int, optional
        Total number of items. If not provided, will be inferred from
        the iterable if possible.
    disable : bool, default False
        If True, disables the progress bar entirely.

    Returns
    -------
    tqdm
        A tqdm progress bar wrapping the iterable.

    Notes
    -----
    For nested progress bars:
    - Outer bars should use position=0, leave=True
    - Inner bars should use position=1 (or higher), leave=False

    Example
    -------
    >>> for file in progress_bar(files, desc="Processing files", position=0):
    ...     for item in progress_bar(items, desc="Items", position=1, leave=False):
    ...         process(item)
    """
    tqdm_class = _get_tqdm_class()

    kwargs = {
        "desc": desc,
        "unit": unit,
        "leave": leave,
        "disable": disable,
    }

    # Only pass position for non-notebook tqdm to avoid issues
    # Jupyter notebook tqdm handles nested bars differently
    if position is not None and not _is_notebook():
        kwargs["position"] = position

    if total is not None:
        kwargs["total"] = total

    return tqdm_class(iterable, **kwargs)
