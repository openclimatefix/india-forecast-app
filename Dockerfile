FROM python:3.11-slim as base

RUN apt-get update
RUN apt-get install -y git

ENV PYTHONFAULTHANDLER=1 \
	PYTHONHASHSEED=random \
	PYTHONUNBUFFERED=1

WORKDIR /app

FROM base as builder

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++

ENV PIP_DEFAULT_TIMEOUT=100 \
	PIP_DISABLE_PIP_VERSION_CHECK=1 \
	PIP_NO_CACHE_DIR=1 \
	POETRY_VERSION=1.8.1

RUN pip install "poetry==$POETRY_VERSION"

RUN python -m venv /venv

COPY pyproject.toml poetry.lock README.md .
RUN . /venv/bin/activate && poetry install --only main --no-root

COPY india_forecast_app ./india_forecast_app
RUN . /venv/bin/activate && poetry build

FROM base as final

ENV PATH="/venv/bin:$PATH"

COPY --from=builder /venv /venv
COPY --from=builder /app/dist .
RUN . /venv/bin/activate && pip install *.whl

COPY nwp.zarr ./nwp.zarr

ENTRYPOINT ["app", "--write-to-db"]