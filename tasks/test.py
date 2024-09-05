"""Task to run tests for this package."""

from invoke import task

from .util import PROJECT_DIR, print_and_run

# NOTE: running "mypy" instead of "mypy --no-incremental" gave hundreds of
#       errors.  SS 7/15/2024
#  Per:
#    https://stackoverflow.com/questions/52189217/use-mypy-with-ruamel-yaml
#  ...it appears this is a known issue.  See:
#    https://github.com/python/mypy/issues/7276
#    https://sourceforge.net/p/ruamel-yaml/tickets/328/


@task(aliases=["mypy"])
def typecheck(ctx):
    """Run mypy typechecking."""
    print_and_run(
        ("mypy --no-incremental"),
        pty=True,
    )

    print("🎉🦆 Type checking passed.")


@task()
def unit(ctx):
    """Run unit tests."""
    print_and_run(
        f"pytest --cov=seaice_ecdr --cov-fail-under 40 -s {PROJECT_DIR}/seaice_ecdr/tests/unit",
        pty=True,
    )


@task()
def integration(ctx):
    """Run integration tests."""
    print_and_run(
        f"pytest -s {PROJECT_DIR}/seaice_ecdr/tests/integration",
        pty=True,
    )


@task()
def regression(ctx):
    """Run regression tests."""
    print_and_run(
        f"pytest -s {PROJECT_DIR}/seaice_ecdr/tests/regression",
        pty=True,
    )


@task()
def pytest(ctx):
    """Run all tests with pytest.

    Includes a code-coverage check.
    """
    print_and_run(
        # "pytest --cov=seaice_ecdr --cov-fail-under 80 -s",
        "pytest --cov=seaice_ecdr --cov-fail-under 70 -s",
        pty=True,
    )


@task(
    pre=[
        typecheck,
        unit,
    ],
)
def ci(ctx):
    """Run tests not requiring access to external data.

    Excludes e.g., regression tests that require access to data on
    NSIDC-specific infrastructure.
    """
    ...


@task(
    pre=[
        typecheck,
        pytest,
    ],
    default=True,
)
def all(ctx):
    """Run all of the tests."""
    ...
