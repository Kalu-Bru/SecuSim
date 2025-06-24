.PHONY: setup install run webapp stats-demo stats test lint format typecheck clean

setup: install
	@echo "Setting up the project..."
	@mkdir -p data reports/figures
	@echo "Project setup complete. Run 'make run' to execute the comparison."

install:
	@echo "Installing dependencies..."
	poetry install

run:
	@echo "Running securitization comparison..."
	poetry run python scripts/compare.py

webapp:
	@echo "Starting web application..."
	poetry run streamlit run app.py

stats-demo:
	@echo "Running statistical analysis demo..."
	poetry run python scripts/statistical_demo.py

stats:
	@echo "Running full statistical analysis..."
	poetry run python scripts/compare.py --statistical-analysis --n-simulations 1000

test:
	@echo "Running tests..."
	poetry run pytest -v

lint:
	@echo "Running linter..."
	poetry run ruff check .
	poetry run ruff check . --fix

format:
	@echo "Formatting code..."
	poetry run black .

typecheck:
	@echo "Type checking..."
	poetry run mypy .

clean:
	@echo "Cleaning up..."
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 