FROM python:3.11-slim

RUN apt-get update
RUN apt-get install -y git

ENV PYTHONFAULTHANDLER=1 \
	PYTHONHASHSEED=random \
	PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get install -y gdal-bin

ENV PIP_DEFAULT_TIMEOUT=100 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1 \
	POETRY_VERSION=1.7.1

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock README.md .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN poetry install --only main --no-root

COPY india_forecast_app ./india_forecast_app
RUN poetry build
RUN poetry install --only main

COPY nwp.zarr ./nwp.zarr

ENTRYPOINT ["poetry", "run" , "python3", "india_forecast_app/app.py", "--write-to-db"]