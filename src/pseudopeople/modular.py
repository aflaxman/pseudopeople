"""
=========================
   Modular noise API
=========================

First-class, dataset-decoupled noise utilities.

pseudopeople's realistic *column* noise -- typos, OCR errors, phonetic errors,
nicknames, fake-name substitution, and blanking -- is useful on its own, even
when the data did not come from pseudopeople's simulated-population pipeline.
This module exposes that capability as plain functions that operate on an
arbitrary list, :class:`pandas.Series`, or :class:`pandas.DataFrame`, with no
parquet pipeline, no ``Dataset``/``COLUMNS`` schema, and no hand-built
``LayeredConfigTree``.

The functions here do not reimplement any noise math: they reuse the existing
:data:`~pseudopeople.noise_entities.NOISE_TYPES` (the ``ColumnNoiseType``
machinery) directly, building the minimal configuration node and randomness
stream those objects need.

Examples
--------
>>> import pseudopeople as psp
>>> psp.add_noise_to_names(["Robert", "Jennifer", "William"], seed=0)  # doctest: +SKIP
['Robert', 'Jennifer', 'Wiliam']
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd
from layered_config_tree import LayeredConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.utilities import get_randomness_stream

# The column-noise types this public API supports -- everything that is
# meaningful on a free-text/string column without dataset-specific companion
# columns. (Noise types such as copy_from_household_member, swap_month_and_day,
# misreport_age, and the zipcode/digit writers depend on dataset structure or
# specific column semantics and are intentionally not exposed here.)
_SUPPORTED: Dict[str, ColumnNoiseType] = {
    NOISE_TYPES.leave_blank.name: NOISE_TYPES.leave_blank,
    NOISE_TYPES.use_nickname.name: NOISE_TYPES.use_nickname,
    NOISE_TYPES.use_fake_name.name: NOISE_TYPES.use_fake_name,
    NOISE_TYPES.make_phonetic_errors.name: NOISE_TYPES.make_phonetic_errors,
    NOISE_TYPES.make_ocr_errors.name: NOISE_TYPES.make_ocr_errors,
    NOISE_TYPES.make_typos.name: NOISE_TYPES.make_typos,
}

# Noise types whose behavior depends on the logical column name.
_NAME_AWARE = {NOISE_TYPES.use_fake_name.name, NOISE_TYPES.use_nickname.name}
# Column names understood by use_fake_name's internal first/last name pool map
# (see noise_functions.use_fake_names).
_FAKE_NAME_COLUMNS = {
    "first_name",
    "middle_name",
    "last_name",
    "spouse_first_name",
    "spouse_last_name",
}

# Default ordering for name noising: structural substitutions first, then
# character-level corruption (mirrors NOISE_TYPES application order).
DEFAULT_NAME_NOISE: Tuple[str, ...] = (
    NOISE_TYPES.use_nickname.name,
    NOISE_TYPES.make_phonetic_errors.name,
    NOISE_TYPES.make_ocr_errors.name,
    NOISE_TYPES.make_typos.name,
)

_NAME_TYPE_TO_COLUMN = {
    "first": "first_name",
    "middle": "middle_name",
    "last": "last_name",
}


def _resolve_noise_type(
    noise_type: Union[str, ColumnNoiseType],
) -> Tuple[str, ColumnNoiseType]:
    if isinstance(noise_type, ColumnNoiseType):
        return noise_type.name, noise_type
    if noise_type in _SUPPORTED:
        return noise_type, _SUPPORTED[noise_type]
    raise ValueError(
        f"Unknown or unsupported noise type {noise_type!r}. "
        f"Supported types are: {sorted(_SUPPORTED)}."
    )


def _build_config_node(
    noise_type: ColumnNoiseType,
    cell_probability: Optional[float],
    extra: Optional[Mapping[str, Any]] = None,
) -> LayeredConfigTree:
    """Build the single configuration node a ``ColumnNoiseType`` expects.

    Starts from the noise type's own baseline (its ``probability`` and any
    ``additional_parameters``, e.g. ``token_probability``) so callers only
    override what they care about, then applies the friendly keyword overrides.
    This delegates knowledge of which keys exist to ``NOISE_TYPES`` rather than
    duplicating it here.
    """
    node: Dict[str, Any] = {}
    if noise_type.probability is not None:
        node[Keys.CELL_PROBABILITY] = noise_type.probability
    if noise_type.additional_parameters:
        node.update(noise_type.additional_parameters)
    if cell_probability is not None:
        node[Keys.CELL_PROBABILITY] = cell_probability
    if extra:
        node.update(extra)
    return LayeredConfigTree(node)


def noise_column(
    data: Union[Sequence, pd.Series],
    noise_type: Union[str, ColumnNoiseType],
    *,
    cell_probability: float = 0.01,
    token_probability: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
    seed: Any = 0,
    column_name: Optional[str] = None,
) -> pd.Series:
    """Apply a single column-noise type to a one-dimensional collection of values.

    :param data:
        A list, tuple, :class:`numpy.ndarray`, or :class:`pandas.Series` of
        (mostly string) values.
    :param noise_type:
        The noise type to apply, either by name (e.g. ``"make_typos"``) or as a
        ``ColumnNoiseType`` from :data:`~pseudopeople.noise_entities.NOISE_TYPES`.
        Supported names: ``leave_blank``, ``use_nickname``, ``use_fake_name``,
        ``make_phonetic_errors``, ``make_ocr_errors``, ``make_typos``.
    :param cell_probability:
        The per-cell probability that a value is selected for noising.
    :param token_probability:
        For token-level noises (typos / OCR / phonetic), the per-eligible-token
        corruption probability. Ignored by noises that do not use it.
    :param config:
        An optional dictionary of additional raw configuration keys forwarded
        into the per-noise configuration node (advanced use).
    :param seed:
        A seed for reproducibility (forwarded to the Vivarium randomness stream).
    :param column_name:
        The logical column name. Defaults to ``data.name`` (if a Series) or
        ``"first_name"``. Must be a recognized name column for ``use_fake_name``.

    :return:
        A :class:`pandas.Series` (object dtype) of the same length and index as
        the input. ``NaN`` / empty / non-string cells are passed through
        untouched, since they are never eligible for noise.

    :raises ValueError:
        An unknown ``noise_type`` is provided, or ``use_fake_name`` is requested
        with a ``column_name`` it does not recognize.
    """
    name, noise = _resolve_noise_type(noise_type)

    is_series = isinstance(data, pd.Series)
    series = data.copy() if is_series else pd.Series(list(data))
    original_index = series.index
    original_name = series.name if is_series else None
    series = series.reset_index(drop=True)

    if column_name is None:
        column_name = series.name if series.name is not None else "first_name"
    series.name = column_name

    if name == NOISE_TYPES.use_fake_name.name and column_name not in _FAKE_NAME_COLUMNS:
        raise ValueError(
            f"use_fake_name requires column_name to be one of "
            f"{sorted(_FAKE_NAME_COLUMNS)}; got {column_name!r}."
        )

    extra: Dict[str, Any] = dict(config or {})
    if token_probability is not None:
        extra[Keys.TOKEN_PROBABILITY] = token_probability
    configuration = _build_config_node(noise, cell_probability, extra)

    frame = series.to_frame()
    # Use the noise type name as the randomness key (analogous to dataset_name)
    # so different noise types draw decorrelated randomness.
    randomness = get_randomness_stream(name, seed, frame.index)

    result, _index_noised = noise(
        frame,
        configuration,
        randomness,
        name,
        column_name,
        missingness=None,
    )
    result.index = original_index
    result.name = original_name if is_series else column_name
    return result


def add_noise_to_names(
    names: Union[Sequence, pd.Series],
    *,
    name_type: str = "first",
    noise_types: Iterable[str] = DEFAULT_NAME_NOISE,
    cell_probability: float = 0.05,
    token_probability: float = 0.1,
    config: Optional[Mapping[str, Mapping[str, Any]]] = None,
    seed: Any = 0,
) -> Union[list, pd.Series]:
    """Add pseudopeople's realistic name noise to a list or Series of names.

    A convenience wrapper that applies a sequence of name-appropriate column
    noises (nicknames, phonetic errors, OCR errors, typos by default) to names
    from any source.

    :param names:
        A list or :class:`pandas.Series` of name strings.
    :param name_type:
        ``"first"``, ``"middle"``, or ``"last"``. Selects the correct
        nickname / fake-name behavior (``"first"`` and ``"middle"`` share the
        first-name pool).
    :param noise_types:
        An ordered iterable of noise type names to apply in sequence. Defaults
        to nickname -> phonetic -> OCR -> typos. Add ``"use_fake_name"`` or
        ``"leave_blank"`` if desired.
    :param cell_probability:
        The default per-cell probability for each noise (override per type via
        ``config``).
    :param token_probability:
        The default per-token probability for token-level noises.
    :param config:
        Optional ``{noise_type: {key: value}}`` per-type overrides, e.g.
        ``{"make_typos": {"cell_probability": 0.2, "token_probability": 0.3}}``.
    :param seed:
        A seed for reproducibility.

    :return:
        The noised names, as the same container type as the input (a list in
        returns a list; a Series in returns a Series with the original index
        and name).

    :raises ValueError:
        An invalid ``name_type`` or an unknown noise type is provided.
    """
    column_name = _NAME_TYPE_TO_COLUMN.get(name_type)
    if column_name is None:
        raise ValueError(
            f"name_type must be 'first', 'middle', or 'last'; got {name_type!r}."
        )

    is_series = isinstance(names, pd.Series)
    series = names.copy() if is_series else pd.Series(list(names))
    current = series.reset_index(drop=True)
    current.name = column_name

    per_type_cfg = config or {}
    for noise_name in noise_types:
        overrides = dict(per_type_cfg.get(noise_name, {}))
        cell = overrides.pop(Keys.CELL_PROBABILITY, cell_probability)
        token = overrides.pop(Keys.TOKEN_PROBABILITY, token_probability)
        # Re-seed deterministically per noise type so the types decorrelate
        # while the whole pipeline stays reproducible for a given ``seed``.
        current = noise_column(
            current,
            noise_name,
            cell_probability=cell,
            token_probability=token,
            config=overrides or None,
            seed=f"{seed}_{noise_name}",
            column_name=column_name,
        )

    if is_series:
        current.index = names.index
        current.name = names.name
        return current
    return list(current)


def noise_dataframe(
    df: pd.DataFrame,
    noise_spec: Mapping[str, Iterable[Union[str, Tuple[str, Mapping[str, Any]]]]],
    *,
    cell_probability: float = 0.05,
    token_probability: float = 0.1,
    seed: Any = 0,
) -> pd.DataFrame:
    """Apply column noise to selected columns of an arbitrary DataFrame.

    :param df:
        The input DataFrame. It is not modified; a noised copy is returned.
    :param noise_spec:
        A mapping ``{column_name: [noise_type, ...]}`` or
        ``{column_name: [(noise_type, {overrides}), ...]}``. Each listed column
        is noised independently, composing its noise types in the given order.
    :param cell_probability:
        The default per-cell probability for each noise.
    :param token_probability:
        The default per-token probability for token-level noises.
    :param seed:
        A seed for reproducibility.

    :return:
        A copy of ``df`` with the specified columns noised.

    :raises KeyError:
        A column in ``noise_spec`` is not present in ``df``.
    :raises ValueError:
        An unknown noise type is provided.
    """
    out = df.copy()
    for column, specs in noise_spec.items():
        if column not in out.columns:
            raise KeyError(f"Column {column!r} is not in the DataFrame.")
        series = out[column]
        for spec in specs:
            if isinstance(spec, tuple):
                noise_name, overrides = spec
                overrides = dict(overrides)
            else:
                noise_name, overrides = spec, {}
            cell = overrides.pop(Keys.CELL_PROBABILITY, cell_probability)
            token = overrides.pop(Keys.TOKEN_PROBABILITY, token_probability)
            series = noise_column(
                series,
                noise_name,
                cell_probability=cell,
                token_probability=token,
                config=overrides or None,
                seed=f"{seed}_{column}_{noise_name}",
                column_name=column if column in _FAKE_NAME_COLUMNS else None,
            )
        out[column] = series
    return out
