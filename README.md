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


This example invokes app.py and passes the help flag
```
DB_URL={DB_URL} poetry run app --help
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

Run the image (this example invokes app.py and passes the help flag)
```
docker run -it --rm ocf/india-forecast-app --help
```
