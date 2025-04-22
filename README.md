<h1 align="center">India-Forecast-App </h1>
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![ease of contribution: hard](https://img.shields.io/badge/ease%20of%20contribution:%20hard-bb2629)](https://github.com/openclimatefix/ocf-meta-repo?tab=readme-ov-file#how-easy-is-it-to-get-involved)

Runs wind and PV forecasts for India and saves to database

## The model

The ML model is from [PVnet](https://github.com/openclimatefix/PVNet) and uses [ocf_datapipes](https://github.com/openclimatefix/ocf_datapipes) for the data processing
For both Wind and Solar we use ECMWF data and predict 48 hours into the future. 

### Wind

The latest change is to use a patch size of 84x84, an increase from 64x64. 
This is to allow for more context in the image and to allow for the model to learn more about the wind patterns.

The validation error is ~ 7.15% (normalized using the maximum wind generation)

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

### Adjuster

The Adjuster model improves forecast accuracy by learning from recent prediction errors. Here's how it works:

1. For each forecast, it analyzes the Mean Error (ME) from forecasts made at the same hour over the past 7 days
2. It calculates the average error for each forecast horizon (e.g., 1-hour ahead, 2-hours ahead, etc.)
3. It then adjusts the current forecast by subtracting these systematic errors

**Real-world example:**
If our ML model consistently under-predicts solar generation by 50kW during sunny mornings (positive ME), the Adjuster will add 50kW to future morning forecasts. Conversely, if it over-predicts evening wind generation by 30kW (negative ME), the Adjuster will subtract 30kW from future evening forecasts.

**Key features:**
- Time-specific: Adjustments depend on the time of day and forecast horizon
- Safety limits: Adjustments are capped at 10% of site capacity to prevent extreme corrections
- Special handling for solar: Ensures zero generation during nighttime

This approach significantly reduces systematic errors and improves overall forecast accuracy.

| Without Adjuster | With Adjuster |
|------------------|---------------|
| Systematic errors persist | Learns from recent patterns |
| Fixed model behavior | Adapts to changing conditions |
| Higher overall error | Reduced forecast error |

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

⚠️ Note: one test for the AD model is skipped locally unless the HF token is set, this HF token can be found in AWS Secret Manager under {environment}/huggingface/token and then can be set via export HUGGINGFACE_TOKEN={token_value} in the repo to run the additional test. In CI tests this secret is set so the test will run there.

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

To save batches, you need to set the `SAVE_BATCHES_DIR` environment variable to directory. 
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

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/priyanshubajaj"><img src="https://avatars.githubusercontent.com/u/58442385?v=4?s=100" width="100px;" alt="priyanshubajaj"/><br /><sub><b>priyanshubajaj</b></sub></a><br /><a href="https://github.com/openclimatefix/india-forecast-app/commits?author=priyanshubajaj" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/india-forecast-app/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Dakshbir"><img src="https://avatars.githubusercontent.com/u/144359831?v=4?s=100" width="100px;" alt="Dakshbir"/><br /><sub><b>Dakshbir</b></sub></a><br /><a href="https://github.com/openclimatefix/india-forecast-app/commits?author=Dakshbir" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MAYANK12SHARMA"><img src="https://avatars.githubusercontent.com/u/145884197?v=4?s=100" width="100px;" alt="MAYANK SHARMA"/><br /><sub><b>MAYANK SHARMA</b></sub></a><br /><a href="https://github.com/openclimatefix/india-forecast-app/commits?author=MAYANK12SHARMA" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Troubleshooting

### Poetry Installation Issues

**Problem**: `poetry install` fails with dependency conflicts
**Solution**: Try updating Poetry first with `pip install --upgrade poetry`, then run `poetry update` followed by `poetry install`

**Problem**: Package installation errors
**Solution**: Check your Python version matches the one specified in `pyproject.toml`. You can use `poetry env use python3.x` to set the correct version.

### Docker Database Connection Issues

**Problem**: Container can't connect to local database with "connection refused" error
**Solution**: If using localhost in your DB_URL, replace it with `host.docker.internal` when running in Docker

**Problem**: Database authentication failures
**Solution**: Verify your DB_URL format is correct: `postgresql://username:password@hostname:port/database`

### Model Loading Issues

**Problem**: "Failed to load model" errors
**Solution**: Ensure your HUGGINGFACE_TOKEN environment variable is set correctly. The token can be found in AWS Secret Manager under {environment}/huggingface/token.

**Problem**: Out of memory errors when loading models
**Solution**: Ensure your system has sufficient RAM, or consider using a smaller model variant.
