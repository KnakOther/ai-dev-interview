#!/bin/bash
# Save current working directory
ORIGINAL_DIR=$(pwd)

# Go to the directory of the script (ie ai package root)
cd "$(dirname "$0")"

# Run checks
poetry run pytest tests
poetry run ruff check --fix src tests lambda_handler.py
poetry run ruff format src tests lambda_handler.py
poetry run mypy src tests lambda_handler.py

# Return to original directory
cd "$ORIGINAL_DIR"
