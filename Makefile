#
# This mostly contains shortcut for multi-command steps.
#
SRC = india_forecast_app scripts tests

.PHONY: lint
lint:
	ruff check $(SRC)

.PHONY: format
format:
	ruff check --fix $(SRC)

.PHONY: test
test:
	pytest tests --disable-warnings

.PHONY: docker.build
docker.build:
	docker build -t ocf/india-forecast-app .
