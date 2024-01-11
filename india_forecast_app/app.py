import click

@click.command()
@click.option("--site", help="Site ID")
def app(site):
	"""Runs the forecast for a given site"""
	print(f"Running forecast for site: {site}")

if __name__ == "__main__":
	app()