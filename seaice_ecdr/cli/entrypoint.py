"""entrypoint.py  Contains click commands fo seaice_ecdr."""
import click

from seaice_ecdr.complete_daily_ecdr import cli as complete_daily_cli
from seaice_ecdr.daily_aggregate import cli as daily_aggregate_cli
from seaice_ecdr.initial_daily_ecdr import cli as ecdr_cli
from seaice_ecdr.monthly import cli as monthly_cli
from seaice_ecdr.monthly_aggregate import cli as monthly_aggregate_cli

# TODO: The multiprocess daily invocation causes an error if the
#       temporal_composite_daily interpolation attempts to access
#       a day prior to the start of SMMR (10/25/1978)
# TODO: I think the overwrite flag might not be defaulting to False,
#       which is probably what we want so that we don't regenerate
#       existing files without explicitly asking to do so
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
cli.add_command(complete_daily_cli)
cli.add_command(monthly_cli)
cli.add_command(daily_aggregate_cli)
cli.add_command(monthly_aggregate_cli)
cli.add_command(validation_cli)
cli.add_command(multiprocess_daily_cli)

if __name__ == "__main__":
    cli()
