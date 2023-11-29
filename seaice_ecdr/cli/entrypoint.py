"""entrypoint.py  Contains click commands fo seaice_ecdr."""
import click

from seaice_ecdr.complete_daily_ecdr import cli as complete_daily_cli
from seaice_ecdr.daily_aggregate import cli as daily_aggregate_cli
from seaice_ecdr.initial_daily_ecdr import cli as ecdr_cli
from seaice_ecdr.monthly import cli as monthly_cli
from seaice_ecdr.monthly_aggregate import cli as monthly_aggregate_cli
from seaice_ecdr.nrt import nrt_cli
from seaice_ecdr.temporal_composite_daily import cli as tiecdr_cli


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


if __name__ == "__main__":
    cli()
