"""entrypoint.py  Contains click commands fo seaice_ecdr."""

import click

from seaice_ecdr.cli.daily import cli as daily_cli
from seaice_ecdr.cli.monthly import cli as monthly_cli
from seaice_ecdr.daily_aggregate import cli as daily_aggregate_cli
from seaice_ecdr.initial_daily_ecdr import cli as ecdr_cli
from seaice_ecdr.intermediate_daily import cli as intermediate_daily_cli
from seaice_ecdr.intermediate_monthly import cli as intermediate_monthly_cli
from seaice_ecdr.monthly_aggregate import cli as monthly_aggregate_cli
from seaice_ecdr.multiprocess_intermediate_daily import (
    cli as multiprocess_intermediate_daily_cli,
)
from seaice_ecdr.nrt import nrt_cli
from seaice_ecdr.publish_daily import cli as publish_daily_cli
from seaice_ecdr.temporal_composite_daily import cli as tiecdr_cli
from seaice_ecdr.validation import cli as validation_cli


@click.group()
def cli():
    """Run the Sea Ice EDCDR."""
    ...


cli.add_command(ecdr_cli)
cli.add_command(tiecdr_cli)
cli.add_command(nrt_cli)
cli.add_command(intermediate_daily_cli)
cli.add_command(intermediate_monthly_cli)
cli.add_command(monthly_aggregate_cli)
cli.add_command(validation_cli)
cli.add_command(multiprocess_intermediate_daily_cli)
cli.add_command(publish_daily_cli)

# CLIs that ops will use below:
# Generate standard daily files ready for publication:
cli.add_command(daily_cli)
# Generate daily aggregate files by year ready for publication:
cli.add_command(daily_aggregate_cli)
# Generate standard monthly files ready for publication:
cli.add_command(monthly_cli)

if __name__ == "__main__":
    cli()
