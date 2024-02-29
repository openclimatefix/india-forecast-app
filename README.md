<h1 align="center">india-forecast-app</h1>

Runs wind and PV forecasts for India and saves to database

## Install dependencies (requires [poetry](https://python-poetry.org/))

```
poetry install
```

## Linting and formatting

Lint with:
```
make lint
```

Format code with:
```
make format
```

## Running tests

```
make test
```

## Running the app locally
Replace `{DB_URL}` with a postgres DB connection string (see below for setting up a ephemeral local DB)

If testing on a local DB, you may use the following script to seed the the DB with a dummy user, site and site_group. 
```
DB_URL={DB_URL} poetry run seeder
```
⚠️ Note this is a destructive script and will drop all tables before recreating them to ensure a clean slate. DO NOT RUN IN PRODUCTION ENVIRONMENTS

This example runs the application and writes the results to stdout
```
DB_URL={DB_URL} NWP_ZARR_PATH={NWP_ZARR_PATH} poetry run app
```

### Starting a local database using docker

```bash
    docker run \
        -it --rm \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_PASSWORD=postgres \
        -p 54545:5432 postgres:14-alpine \
        postgres
```

The corresponding `DB_URL` will be

`postgresql://postgres:postgres@localhost:54545/postgres`

## Building and running in [Docker](https://www.docker.com/)

Build the Docker image
```
make docker.build
```

Create a container from the image. This example runs the application and writes the results to stdout.\
Replace `{DB_URL}` with a postgres DB connection string.\
*N.B if the database host is `localhost` on the host machine, replace `localhost` with `host.docker.internal` so that docker can access the database from within the container*
```
docker run -it --rm -e DB_URL={DB_URL} -e NWP_ZARR_PATH={NWP_ZARR_PATH} ocf/india-forecast-app
```

## Notes

This repo makes use of PyTorch (`torch` and `torchvision` packages) CPU-only version. In order to support installing PyTorch via poetry for various environments, we specify the exact wheels for each environment in the pyproject.toml file. Some background reading on why this is required can be found here: https://santiagovelez.substack.com/p/how-to-install-torch-cpu-in-poetry?utm_campaign=post&utm_medium=web&triedRedirect=true 
