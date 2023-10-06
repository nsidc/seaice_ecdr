<p align="center">
  <img alt="NSIDC logo" src="https://nsidc.org/themes/custom/nsidc/logo.svg" width="150" />
</p>


# Enhanced SeaIce CDR

Enhanced SeaIce CDR enables the creation of the 12.5km SeaIce CDR


## Level of Support

* This repository is not actively supported by NSIDC but we welcome issue submissions and
  pull requests in order to foster community contribution.

See the [LICENSE](GENERAL) for details on permissions and warranties. Please contact
nsidc@nsidc.org for more information.


## Requirements

The ]seaice_ecdr_vm repository](https://bitbucket.org/nsidc/seaice_ecdr_vm/src/main/) provides the NSIDC VM configuration for this project.


## Installation

Clone the seaice_ecdr_vm repository and check out appropriate branches of pm_icecon and seaice_ecdr to install this package


## Usage

An initial copy of the pm_icecon "cdr" generation can be executed from the VM directory:

~/seaice_ecdr/

using the cli.sh command:

./scripts/cli.sh bootstrap amsr2 --date 2022-08-01 --hemisphere north --output-dir /tmp/ --resolution 12


## Troubleshooting

No specific troubleshooting suggestions are currently available.

## License

See [LICENSE](GENERAL).


## Code of Conduct

See [Code of Conduct](CODE_OF_CONDUCT.md).


## Credit

This content was developed by the National Snow and Ice Data Center with funding from
multiple sources.
