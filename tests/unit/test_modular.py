"""Unit tests for the modular, dataset-decoupled noise API."""

import numpy as np
import pandas as pd
import pytest

import pseudopeople as psp
from pseudopeople.modular import add_noise_to_names, noise_column, noise_dataframe
from pseudopeople.noise_scaling import load_nicknames_data

NAMES = [
    "William",
    "Elizabeth",
    "Robert",
    "Jennifer",
    "Michael",
    "Katherine",
    "Christopher",
    "Margaret",
    "Joseph",
    "Patricia",
]


def test_public_api_exports():
    # The modular functions are re-exported from the top-level package.
    assert psp.add_noise_to_names is add_noise_to_names
    assert psp.noise_column is noise_column
    assert psp.noise_dataframe is noise_dataframe


def test_noise_column_preserves_shape_and_index():
    s = pd.Series(NAMES, index=range(100, 100 + len(NAMES)), name="first_name")
    result = noise_column(
        s, "make_typos", cell_probability=0.5, token_probability=0.5, seed=1
    )
    assert isinstance(result, pd.Series)
    assert len(result) == len(s)
    assert list(result.index) == list(s.index)
    assert result.name == "first_name"


def test_noise_column_accepts_list_and_returns_series():
    result = noise_column(
        NAMES, "make_typos", cell_probability=1.0, token_probability=1.0, seed=1
    )
    assert isinstance(result, pd.Series)
    assert len(result) == len(NAMES)


def test_noise_column_is_reproducible():
    a = noise_column(
        NAMES, "make_typos", cell_probability=0.8, token_probability=0.5, seed=42
    )
    b = noise_column(
        NAMES, "make_typos", cell_probability=0.8, token_probability=0.5, seed=42
    )
    pd.testing.assert_series_equal(a, b)


def test_noise_column_seed_changes_output():
    a = noise_column(NAMES, "make_typos", cell_probability=0.8, token_probability=0.5, seed=1)
    b = noise_column(NAMES, "make_typos", cell_probability=0.8, token_probability=0.5, seed=2)
    assert not a.equals(b)


def test_noise_column_passes_through_missing_and_nonstring():
    mixed = pd.Series(["Robert", None, "", "Jennifer", np.nan], name="first_name")
    result = noise_column(
        mixed, "make_typos", cell_probability=1.0, token_probability=1.0, seed=3
    )
    # None / empty / NaN are never eligible -> preserved exactly.
    assert result.iloc[1] is None or pd.isna(result.iloc[1])
    assert result.iloc[2] == ""
    assert pd.isna(result.iloc[4])
    # The real strings are eligible and (with p=1) get corrupted.
    assert result.iloc[0] != "Robert"
    assert result.iloc[3] != "Jennifer"


def test_noise_column_unknown_type_raises():
    with pytest.raises(ValueError, match="unsupported noise type"):
        noise_column(NAMES, "not_a_real_noise", seed=0)


def test_leave_blank_blanks_selected_cells():
    result = noise_column(NAMES, "leave_blank", cell_probability=1.0, seed=0)
    assert result.isna().all()


def test_use_nickname_replaces_known_names():
    nicknames = load_nicknames_data()
    robert_nicks = set(nicknames.loc["Robert"].dropna())
    result = noise_column(
        ["Robert"] * 10,
        "use_nickname",
        cell_probability=1.0,
        seed=0,
        column_name="first_name",
    )
    # Every "Robert" was eligible and replaced with one of its real nicknames.
    assert (result != "Robert").all()
    assert set(result).issubset(robert_nicks)


def test_use_fake_name_requires_valid_column_name():
    with pytest.raises(ValueError, match="use_fake_name requires column_name"):
        noise_column(NAMES, "use_fake_name", cell_probability=1.0, seed=0, column_name="nope")


def test_add_noise_to_names_list_in_list_out():
    out = add_noise_to_names(NAMES, name_type="first", cell_probability=0.5, seed=7)
    assert isinstance(out, list)
    assert len(out) == len(NAMES)


def test_add_noise_to_names_series_preserves_index_and_name():
    s = pd.Series(NAMES, index=range(5, 5 + len(NAMES)), name="given_name")
    out = add_noise_to_names(s, name_type="first", cell_probability=0.5, seed=7)
    assert isinstance(out, pd.Series)
    assert list(out.index) == list(s.index)
    assert out.name == "given_name"


def test_add_noise_to_names_reproducible():
    a = add_noise_to_names(NAMES, name_type="first", cell_probability=0.5, seed=7)
    b = add_noise_to_names(NAMES, name_type="first", cell_probability=0.5, seed=7)
    assert a == b


def test_add_noise_to_names_actually_noises():
    # With high probabilities, most names should change.
    out = add_noise_to_names(
        NAMES, name_type="first", cell_probability=0.9, token_probability=0.9, seed=7
    )
    changed = sum(o != n for o, n in zip(NAMES, out))
    assert changed >= len(NAMES) // 2


def test_add_noise_to_names_bad_name_type():
    with pytest.raises(ValueError, match="name_type"):
        add_noise_to_names(NAMES, name_type="surname")


def test_noise_dataframe_noises_only_specified_columns():
    df = pd.DataFrame(
        {
            "first_name": ["Robert", "Jennifer", "William", "Susan"],
            "last_name": ["Smith", "Johnson", "Garcia", "Lee"],
            "id": [1, 2, 3, 4],
        }
    )
    out = noise_dataframe(
        df,
        {
            "first_name": ["make_typos"],
            "last_name": [("make_phonetic_errors", {"token_probability": 0.6})],
        },
        cell_probability=1.0,
        token_probability=0.8,
        seed=99,
    )
    # Untouched column is identical; input frame is not mutated.
    assert out["id"].equals(df["id"])
    assert df["first_name"].tolist() == ["Robert", "Jennifer", "William", "Susan"]
    assert out.shape == df.shape


def test_noise_dataframe_missing_column_raises():
    df = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(KeyError):
        noise_dataframe(df, {"b": ["make_typos"]})
