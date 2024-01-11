#
# This mostly contains shortcut for multi-command steps.
#
SRC = india_forecast_app tests

.PHONY: lint
lint:
	poetry run ruff $(SRC)
	poetry run black --check $(SRC)
	poetry run isort --check $(SRC)


.PHONY: format
format:
	poetry run ruff --fix $(SRC)
	poetry run black $(SRC)
	poetry run isort $(SRC)

.PHONY: test
test:
	poetry run pytest tests
	
.PHONY: docker.build
docker.build:
	docker build -t ocf/india-forecast-app .
