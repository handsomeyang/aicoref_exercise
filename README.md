# Aicoref exercise
[![CI](https://github.com/handsomeyang/aicoref_exercise/actions/workflows/ci.yml/badge.svg)](https://github.com/handsomeyang/aicoref_exercise/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/handsomeyang/aicoref_exercise/graph/badge.svg?token=AE7KA1S7BL)](https://codecov.io/gh/handsomeyang/aicoref_exercise)

## Approach
1. Experimental phase (`notebooks`): data exploration, model validation, feature engineering experiments, etc.
2. Implementation phase (`src`): integrate validated pipeline/code snippets into application structure, improve and iterate.

## Tool Stack
* Pandas
* scikit-learn
* FastAPI
* Google Gemini
* PyCharm + Google Gemini Code Assist
* uv
* pre-commit (ruff, bandit, mypy, gitleaks)
* GitHub Action (ruff, bandit, mypy, gitleaks, pytest)

## Setup
1. Install [uv](https://docs.astral.sh/uv/) on your system.
2. Clone this repo.
3. CD into project root and run `uv sync`.
4. Create a `data` folder under project root, and copy `dataset.csv` there.

## Usage
### Model training
Run `uv run train` at project root, which by default will:
1. Split `dataset.csv` into training and testing datasets.
2. Encode binary and categorical features, and standardise numeric features.
3. Run hyperparameter search via CV on a XGBoost classifier.
4. Evaluate the best XGBoost model on the testing set.
5. Save training artifacts (data processing + best model pipeline, features used in training) in the `artifacts` folder.

Note:
* For supported arguments, run `uv run train --help`.

### Model serving
Run `uv run serve` at project root, which by default will:
1. Load the data + best model pipeline persisted from the training stage (a trained pipeline has been provided in the `artifacts` folder for convenience).
2. Spin up a FastAPI server and expose model inference via the `/predict` endpoint.

Note:
* `uv run serve` runs the server in production mode. For dev mode (hot reloading), run `uv run serve --dev`.
* For supported arguments, run `uv run serve --help`.

### Model querying
Run `uv run query` at project root, which by default will:
1. Randomly sample a data point from `dataset.csv`.
2. Send the data to the `/predict` endpoint.
3. Print out prediction result.

Note:
* To predict on new data, use the `--data` argument.
* For supported arguments, run `uv run query --help`.

## TODO
* Improve documentation (docstrings + comments).
* Improve test coverage.
* Improve logging (structured logs).
* Add model monitoring.
* Model optimisation.

## Use of AI
In the development of this project, AI is used in the following ways:
1. Google Gemini is consulted to suggest best practices, validate ideas, troubleshoot errors, etc.
2. Google Gemini Code Assist is used to generate a few unit tests to validate the CI pipeline.
