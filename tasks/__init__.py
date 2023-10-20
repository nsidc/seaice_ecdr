"""Add format and test to collections."""
from invoke import Collection

from . import test

ns = Collection()
ns.add_collection(test)  # type: ignore
