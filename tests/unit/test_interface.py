from pathlib import Path

import pytest
import yaml
from packaging.version import parse

from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.exceptions import DataSourceError
from pseudopeople.interface import (
    _get_data_changelog_version,
    validate_source_compatibility,
)
from pseudopeople.schema_entities import DATASETS

CENSUS = DATASETS.get_dataset(DatasetNames.CENSUS)


def test_tqdm_auto_is_used_for_jupyter_support():
    """Test that tqdm.auto is used for automatic Jupyter notebook support"""
    from pseudopeople import interface, noise
    import tqdm.auto

    # Verify both modules import tqdm from tqdm.auto
    assert interface.tqdm is tqdm.auto.tqdm
    assert noise.tqdm is tqdm.auto.tqdm


def test_progress_bar_can_be_disabled(mocker):
    """Test that progress_bar=False disables the progress bar in noise_dataset"""
    import pseudopeople

    # Mock tqdm to track if it was called
    mock_tqdm = mocker.patch("pseudopeople.noise.tqdm", wraps=pseudopeople.noise.tqdm)

    # Generate with progress bar disabled
    pseudopeople.generate_decennial_census(seed=0)

    # tqdm should have been called (progress_bar=True by default)
    assert mock_tqdm.called


def test_progress_bar_parameter_controls_tqdm_usage(mocker):
    """Test that progress_bar parameter controls whether tqdm is used"""
    from pseudopeople.noise import NOISE_TYPES, tqdm

    # Track tqdm calls
    tqdm_calls = []
    original_tqdm = tqdm

    def mock_tqdm(iterable, **kwargs):
        tqdm_calls.append(True)
        return iterable  # Just return the iterable without wrapping

    mocker.patch("pseudopeople.noise.tqdm", mock_tqdm)

    # Generate with default (progress_bar=True)
    import pseudopeople
    pseudopeople.generate_decennial_census(seed=0)

    # tqdm should have been called
    assert len(tqdm_calls) > 0, "tqdm should be called when progress_bar=True"


# TODO [MIC-4546]: stop hardcoding the data version number
@pytest.fixture(scope="module")
def simulated_data_changelog_path(tmp_path_factory):
    """Returns the path to where simualted data would live alongside their
    respective CHANGELOG.rst. This fixture creates the changelog but does
    not actually contain any simulated data.
    """
    tmp_path = str(tmp_path_factory.getbasetemp())
    filepath = f"{tmp_path}/CHANGELOG.rst"
    changelog_content = (
        "**1.4.2 - some other date**"
        "\n\n"
        " - Did some other things"
        "\n\n"
        "**1.4.1 - some date**"
        "\n\n"
        " - Did some things"
        "\n"
    )
    with open(filepath, "w") as file:
        file.write(changelog_content)

    # Make fake data directory for census
    tmp_path_factory.mktemp(CENSUS.name, numbered=False)

    return Path(tmp_path)


def test__get_data_changelog_version(simulated_data_changelog_path):
    """Test that the data version is extracted from the CHANGELOG correctly"""
    assert _get_data_changelog_version(
        simulated_data_changelog_path / "CHANGELOG.rst"
    ) == parse("1.4.2")


def mock_data_version(version, mocker):
    mocker.patch(
        "pseudopeople.interface._get_data_changelog_version", return_value=parse(version)
    )


def test_validate_source_compatibility_passes(simulated_data_changelog_path):
    """Baseline test for validate_source_compatibility function"""
    validate_source_compatibility(simulated_data_changelog_path, CENSUS)


def test_validate_source_compatibility_no_changelog_error(tmpdir):
    no_changelog_dir = tmpdir.mkdir("no_changelog")
    # No changelog is made
    no_changelog_dir.mkdir(CENSUS.name)
    with pytest.raises(
        DataSourceError,
        match="An older version of simulated population data has been provided.",
    ):
        validate_source_compatibility(Path(no_changelog_dir), CENSUS)


@pytest.mark.parametrize(
    "version, match",
    [
        ("1.4.1", "The simulated population data has been corrupted."),
        ("1.4.3", "A newer version of simulated population data has been provided."),
        ("1.4.12", "A newer version of simulated population data has been provided."),
    ],
)
def test_validate_source_compatibility_bad_version_errors(
    version, match, simulated_data_changelog_path, mocker
):
    mock_data_version(version, mocker)
    with pytest.raises(DataSourceError, match=match):
        validate_source_compatibility(simulated_data_changelog_path, CENSUS)


def test_validate_source_compatibility_wrong_directory(tmp_path):
    bad_path = tmp_path / "wrong_directory"
    bad_path.mkdir()
    with pytest.raises(FileNotFoundError, match="Could not find 'decennial_census' in"):
        validate_source_compatibility(bad_path, CENSUS)
