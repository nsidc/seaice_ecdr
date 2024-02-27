from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


@cache
def _get_cached_tmpdir():
    tmpdir = TemporaryDirectory()

    return tmpdir


@pytest.fixture(scope="session")
def base_output_dir_test_path():
    """Session-scoped fixture providing temporary dir representing ECDR data dir."""
    tmpdir = _get_cached_tmpdir()
    tmppath = Path(tmpdir.name)

    yield tmppath

    tmpdir.cleanup()
