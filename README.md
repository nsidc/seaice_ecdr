<p float="left" align="center">
    <img alt="NSIDC logo" src="https://nsidc.org/themes/custom/nsidc/logo.svg" height="150" />
    <img alt="NOAA@NSIDC logo" src="https://nsidc.org/sites/default/files/images/Logo/noaa_at_nsidc.png" height="150" />
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

`seaice_ecdr` is primarily interacted with through it's CLI. To utilize the CLI,
use the provided `scripts/cli.sh`:

```
$ ./scripts/cli.sh  --help
Usage: python -m seaice_ecdr.cli.entrypoint [OPTIONS] COMMAND [ARGS]...

  Run the Sea Ice EDCDR.

Options:
  --help  Show this message and exit.

Commands:
  idecdr  Run the initial daily ECDR algorithm with AMSR2 data.
  nrt     Run NRT Sea Ice ECDR.
```

### Logging

By default, logs are written to disk at
`/share/apps/G02202_V5/{ECDR_PRODUCT_VERSION}/{YYYY-MM-DD}.log`. Up to 31 of
these logs can exist at once (older log files get removed).

During large re-processing efforts, it may be desirable to temporarily disable
logging to improve processing speed and reduce disk usage. To do so, set the
`DISABLE_FILE_LOGGING` envvar to `TRUE`.

```
export DISABLE_FILE_LOGGING=TRUE
```

## Development/contributing

See [doc/development.md](doc/development.md) for more information.

## License

See [LICENSE](LICENSE).


## Code of Conduct

See [Code of Conduct](CODE_OF_CONDUCT.md).


## Credit

This software was developed by the National Snow and Ice Data Center with
funding from NOAA.
