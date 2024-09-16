"""entrypoint.py  Contains click commands fo seaice_ecdr."""

import click

# TODO: The daily-aggregate processing is very parallelizable because
#       each year is indendent of every other year.  It could be
#       implemented with multi-processing to speed up production
#       on a multi-core machine.  Perhaps as a cmdline arg to this
#       CLI API?
from seaice_ecdr.daily_aggregate import cli as daily_aggregate_cli
from seaice_ecdr.initial_daily_ecdr import cli as ecdr_cli
from seaice_ecdr.intermediate_daily import cli as intermediate_daily_cli
from seaice_ecdr.monthly import cli as monthly_cli
from seaice_ecdr.monthly_aggregate import cli as monthly_aggregate_cli
from seaice_ecdr.multiprocess_daily import cli as multiprocess_daily_cli
from seaice_ecdr.nrt import nrt_cli
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
cli.add_command(monthly_cli)
cli.add_command(daily_aggregate_cli)
cli.add_command(monthly_aggregate_cli)
cli.add_command(validation_cli)
cli.add_command(multiprocess_daily_cli)

if __name__ == "__main__":
    cli()
