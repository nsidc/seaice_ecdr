"""Task to run tests for this package."""
from invoke import task

from .util import PROJECT_DIR, print_and_run


@task(aliases=["mypy"])
def typecheck(ctx):
    """Run mypy typechecking."""
    print_and_run(
        ("mypy"),
        pty=True,
    )

    print("🎉🦆 Type checking passed.")


@task()
def unit(ctx):
    """Run unit tests."""
    print_and_run(
        f"PYTHONPATH={PROJECT_DIR} pytest -s {PROJECT_DIR}/seaice_ecdr/tests/unit",
        pty=True,
    )


@task(
    pre=[
        typecheck,
        unit,
    ],
    default=True,
)
def all(ctx):
    """Run all of the tests."""
    ...
