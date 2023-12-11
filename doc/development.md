While developing at NSIDC, the workflow is:

### For developers at NSIDC

For developers at NSIDC, the [seaice_ecdr_vm
repository](https://bitbucket.org/nsidc/seaice_ecdr_vm/src/main/) provides the
NSIDC VM configuration for this project.

`pm_icecon`, `seaice_ecdr`, and `pm_tb_data` will be checked out to
`/home/vagrant/{project_name}`.

### Adding dependencies

To add new dependencies to this project, update the `environment.yml` file with
the new dependency. Then update your conda environment:

```
$ mamba env update
```

Once the conda environment has been updated, lock the environment using `conda-lock`:

```
$ conda-lock -p linux-64
```

Commit the changes for the `environment.yml` and the `conda-lock.yml` files.


### Running tests/CI

#### Linting / formatting
This project uses [pre-commit](https://pre-commit.com/) to run pre-commit hooks
that check and format this project's code for stylistic consistency (using
`ruff` and `black`) .

The pre-commit configuration for this project can be found in
`.pre-commit-config.yaml`. Configuration for specific tools (e.g., `mypy`) is
given in the included `pyproject.toml`.

For more information about using `pre-commit`, please sese the [Scientific
Python Library Development Guide's section on
pre-commit](https://learn.scientific-python.org/development/guides/gha-basic/#pre-commit).

To install pre-commit to run checks for each commit you make:

```
$ pre-commit install
```

To manually run the pre-commit hooks without a commit:

```
$ pre-commit run --all-files
```

#### Running unit tests

Use `pytest` to run unit tests:

```
$ python -m pytest
```

Alternatively, use `scripts/run_ecdr_pytest.sh`.

#### Type-checking

Use `mypy` to run static typechecking

```
$ mypy
```

## Running tests / generating sample data

Can run unit and integration tests -- or manual subsets of -- with:
    
    ./scripts/run_ecdr_pytest.sh

Create an Initial Daily ECDR file with e.g.,:

    ./scripts/cli.sh idecdr --date 2021-04-05 --hemisphere north --resolution 12 --output-dir /tmp/


## Generating ancillary data

Some ancillary data (e.g., surface masks) get created once from input data
sources.

To create the surface/geo mask netcdf files (containing e.g., `surface_type` and
`polehole_bitmask` variables), see
[scripts/surface_geo_masks/README.md](scripts/surface_geo_masks/README.md)
