#
# This mostly contains shortcut for multi-command steps.
#
SRC = india_forecast_app scripts tests

.PHONY: lint
lint:
	poetry run ruff $(SRC)

.PHONY: format
format:
	poetry run ruff --fix $(SRC)

.PHONY: test
test:
	poetry run pytest tests --disable-warnings
	
.PHONY: docker.build
docker.build:
	docker build -t ocf/india-forecast-app .
