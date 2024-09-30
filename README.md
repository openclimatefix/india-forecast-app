<h1 align="center">india-forecast-app</h1>

Runs wind and PV forecasts for India and saves to database

## The model

The ML model is from [PVnet](https://github.com/openclimatefix/PVNet) and uses [ocf_datapipes](https://github.com/openclimatefix/ocf_datapipes) for the data processing
For both Wind and Solar we use ECMWF data and predict 48 hours into the future. 

### Wind

The latest change is to use a patch size of 84x84, an increase from 64x64. 
This is to allow for more context in the image and to allow for the model to learn more about the wind patterns.

The validation error is ~ 7.15% (normalized using the maximum wind generation)

- wandb [link](https://wandb.ai/openclimatefix/india/runs/xdlew7ib)
- hugging face [link](https://huggingface.co/openclimatefix/windnet_india)

The weather variables are currently
- t2m
- u10
- u100
- u200
- v10
- v100
- v200

We add some model smoothing
- feathering [feathering](https://github.com/openclimatefix/india-forecast-app/blob/main/india_forecast_app/models/pvnet/model.py#L131) close to current generation: 
- [smoothing](https://github.com/openclimatefix/india-forecast-app/blob/main/india_forecast_app/models/pvnet/model.py#L188) over 1 hour rolling window


### PV

The validation error is ~ 2.28% (normalized using the maximum solar generation)

- wandb [link](https://wandb.ai/openclimatefix/pvnet_india2.1/runs/o4xpvzrc)
- hugging face [link](https://huggingface.co/openclimatefix/pvnet_india)

The weather variables are
- hcc
- lcc
- mcc
- prate
- sde
- sr
- t2m
- tcc
- u10
- v10
- dlwrf
- dswrf

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

⚠️ Note: one test for the AD model is skipped locally unless the HF token is set, this HF token can be found in AWS Secret Manager under {enviornment}/huggingface/token and then can be set via export HUGGINGFACE_TOKEN={token_value} in the repo to run the additional test. 

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
