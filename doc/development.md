While developing at NSIDC, the workflow is:

### For developers at NSIDC

For developers at NSIDC, the [seaice_ecdr_vm
repository](https://bitbucket.org/nsidc/seaice_ecdr_vm/src/main/) provides the
NSIDC VM configuration for this project.

`pm_icecon`, `seaice_ecdr`, and `pm_tb_data` will be checked out to
`/home/vagrant/{project_name}`.

### Developing all three projects at once

On dev machines, all three repositories are checked out as described above. The
`PYTHONPATH` environment variable is set to point to each of these repositories,
ensuring that changes made to e.g., `pm_icecon` one are reflected when running
code in `seaice_ecdr`.

To coordinate the dependencies requried for all three libraries, the
`scripts/make_dev_environment_yml.py` script can be used to generate a
`dev_environment.yml` that includes the dependencies for all of the
projects. The developer can then update or re-create the `seaice_ecdr` conda
environment using this dev environment file:


```
$ conda deactivate seaice_ecdr
$ conda env remove -n seaice_ecdr
$ python ./scripts/make_dev_environment_yml.py
$ mamba env create -f dev_environment.yml
$ conda activate seaice_ecdr
```

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


## Making a release and updating versions

### Making a code release

Note that, depending on the changes made, a new version of G02202 may also be
appropriate. Please review the secion on "Making a data release" if a new data
release is also needed.

#### Versioning the code

First, appropriately version the code with `bumpversion` (see
[bump-my-version](https://github.com/callowayproject/bump-my-version)).

To bump the specified part of the version:

```
$ bumpversion bump {major|minor|patch}
```

`bumpversion` configuration can be found in the `pyproject.toml`.

#### Updating the CHANGELOG

Each PR should update the CHANGELOG to reflect the version of the ECDR that's
being released/prepared.

#### Releasing a new version

To release a new version of the software, create a tag for the version you wish
to release and push that tag to the GitHub repo.

TODO: on tags, build and push a tagged Docker image and/or conda library. This
code is still in development and a formal release artifact is not currently
being created.


### Making a data release

When making a new release of G02202 based on this code:

* Ensure the `seaice_ecdr.constants.ECDR_PRODUCT_VERSION` has been updated.
* Ensure the ATBD and other supporting documentation have been updated for the
  new release, if necessary.
* Package the code and ancillary data using the
  `scripts/make_archive_for_noaa.py` python-based CLI. This generates a `.zip`
  archive containing the `seaice_ecdr`, `pm_icecon`, and `pm_tb_data`
  repositories and ancillary data. This `.zip` package should then be sent to
  NOAA for archival.


### Replicating CDRv4 results

There are a number of improvements and bug fixes that have been made since the
CDRv4. To produce CDRv4-compatible results with this code, checkout the
`match-cdrv4` tag (commit 87271ec):

```
git checkout match-cdrv4
```
