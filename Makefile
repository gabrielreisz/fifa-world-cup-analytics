.PHONY: install data report evaluate test lint dashboard api clean

install:        ## install the package with dev + app extras
	pip install -e ".[app,dev]"

data:           ## download and cache the raw datasets
	worldcup build-data

report:         ## print headline findings and render figures
	worldcup report

evaluate:       ## back-test the match-outcome model
	worldcup evaluate

test:           ## run the unit test suite
	pytest -m "not integration"

coverage:       ## run tests with coverage and refresh the badge
	pytest -m "not integration" --cov=worldcup --cov-report=term-missing --cov-report=xml
	genbadge coverage -i coverage.xml -o assets/coverage.svg

lint:           ## run ruff
	ruff check src tests

train:          ## train and persist the match predictor
	worldcup train

docker-up:      ## build and run the API + dashboard with docker compose
	docker compose up --build

dashboard:      ## launch the Streamlit dashboard
	streamlit run app/streamlit_app.py

api:            ## launch the FastAPI prediction service
	uvicorn app.api:app --reload

clean:          ## remove caches and generated artifacts
	rm -rf .pytest_cache .ruff_cache **/__pycache__ reports/figures/*.png
