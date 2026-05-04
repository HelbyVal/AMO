#!/bin/bash

set -e

poetry run python data_creation.py
poetry run python model_preprocessing.py
poetry run python model_preparation.py
poetry run python model_testing.py

echo "Pipeline completed successfully"
