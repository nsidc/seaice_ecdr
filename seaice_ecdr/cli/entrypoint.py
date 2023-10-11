"""entrypoint.py  Contains click commands fo seaice_ecdr."""
import click
from pm_icecon.bt.cli import cli as bt_cli
from pm_icecon.cdr import cli as cdr_cli
from pm_icecon.nt.cli import cli as nt_cli

from seaice_ecdr.initial_daily_ecdr import cli as ecdr_cli


@click.group()
def cli():
    """Run the nasateam or bootstrap algorithm."""
    ...


cli.add_command(bt_cli)
cli.add_command(nt_cli)
cli.add_command(cdr_cli)
cli.add_command(ecdr_cli)


if __name__ == "__main__":
    from pm_icecon.cli.entrypoint import cli

    # from seaice_ecdr.cli.entrypoint import cli_ecdr

    cli()
