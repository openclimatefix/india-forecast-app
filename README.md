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

## Building and running in [Docker](https://www.docker.com/)

Build the Docker image
```
make docker.build
```

Run the image (this example invokes app.py and passes the help flag)
```
docker run -it --rm ocf/india-forecast-app --help
```
