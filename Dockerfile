FROM python:3.11-slim

RUN apt-get update
RUN apt-get install -y git

ENV PYTHONFAULTHANDLER=1 \
	PYTHONHASHSEED=random \
	PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++

ENV PIP_DEFAULT_TIMEOUT=100 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1 \
	POETRY_VERSION=1.7.1

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock README.md .
RUN poetry install --no-dev --no-root

COPY india_forecast_app ./india_forecast_app
RUN poetry build
RUN poetry install

COPY nwp.zarr ./nwp.zarr

ENTRYPOINT ["poetry", "run" , "python3", "india_forecast_app/app.py", "--write-to-db"]