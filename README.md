<p float="left" align="center">
    <img alt="NSIDC logo" src="https://nsidc.org/themes/custom/nsidc/logo.svg" height="150" />
    <img alt="NOAA@NSIDC logo" src="https://nsidc.org/sites/default/files/images/Logo/noaa_at_nsidc.png" height="150" />
    <img alt="NASA logo" src="https://gpm.nasa.gov/sites/default/files/document_files/NASA-Logo-Large.png" height="150" />
</p>

# Enhanced Sea Ice CDR

Enhanced SeaIce CDR (ECDR) enables the creation of the 12.5km SeaIce CDR.

Please note that this repository is a work in progress and breaking changes are
to be expected. Initial work on this repository is specific to NSIDC's internal
systems and may not work as expected for external collaborators.


## Level of Support

* This repository is not actively supported by NSIDC but we welcome issue submissions and
  pull requests in order to foster community contribution.

See the [LICENSE](GENERAL) for details on permissions and warranties. Please contact
nsidc@nsidc.org for more information.


## Requirements and installation

This code relies on the python packages defined in the included
`environment.yml` file.

Use [conda](https://docs.conda.io/en/latest/) or
[mamba](https://mamba.readthedocs.io/en/latest/index.html) to install the
requirements:

```
$ conda env create
```

To activate the environment:

```
$ conda activate pm_tb_data
```

## Usage

TODO

## Development/contributing

### For developers at NSIDC

For developers at the NSIDC, the [seaice_ecdr_vm
repository](https://bitbucket.org/nsidc/seaice_ecdr_vm/src/main/) provides the
NSIDC VM configuration for this project.

An initial copy of the pm_icecon "cdr" generation can be executed from the VM directory:

`~/seaice_ecdr/`

using the cli.sh command:

```
./scripts/cli.sh bootstrap amsr2 --date 2022-08-01 --hemisphere north --output-dir /tmp/ --resolution 12
```

### Adding dependencies

To add new dependencies to this project, update the `environment.yml` file with
the new dependency. Then update your conda environment:

```
$ mamba env update
```

Once the conda environment has been updated, lock the environment using `conda-lock`:

```
$ conda-lock
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

## License

See [LICENSE](LICENSE).


## Code of Conduct

See [Code of Conduct](CODE_OF_CONDUCT.md).


## Credit

This software was developed by the National Snow and Ice Data Center with
funding from NASA and NOAA.
