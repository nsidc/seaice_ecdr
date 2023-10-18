"""entrypoint.py  Contains click commands fo seaice_ecdr."""
import click

from seaice_ecdr.initial_daily_ecdr import cli as ecdr_cli


@click.group()
def cli():
    """Run the Sea Ice EDCDR."""
    ...


cli.add_command(ecdr_cli)


if __name__ == "__main__":
    cli()
