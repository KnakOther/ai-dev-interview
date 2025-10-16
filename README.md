# Document Intelligence AI Microservice

This microservice provides AI-powered document analysis capabilities including summarization, entity extraction, and classification.

## Features

- **Document Summarization**: Generate concise summaries with key points extraction
- **Entity Extraction**: Identify people, organizations, locations, dates, and key terms
- **Document Classification**: Categorize documents by topic, type, and sentiment

## Development Environment Setup

Follow these steps to properly set up your development environment.
The recommended setup uses a Docker development container (Dev Container in VSCode) and the VSCode IDE. For dependency and package management, we use [poetry](https://python-poetry.org/).

### 1. Set AWS Credentials

Using the AWS CLI, set your credentials for the AWS testing environment.

```sh
aws sso login --profile developer-353132507445
```

Note: You can find your AWS authentication profile at `~/.aws/credentials`

### 2. Install Required VSCode Extensions

Ensure the following extensions are installed in VSCode:

* Dev Containers
* Python
* Python Debugger
* Pylance

### 3. Create a `.env` file

Some secrets are unique to your environment and should be stored locally in `.env`.

1. Copy the template:
```
cp .env.dist .env
```
Fill in the values with your own secrets.

Note: Any variables defined in devcontainer.json under containerEnv will override values in your .env file.

### 4. Open the project from the Dev Container

First, open VS Code in the project directory.
Reach VSCode's Command Palette using `⇧Shift` + `⌘Cmd` + `P` and select **Dev Containers: Reopen Folder in Container**.

The Dev Container will manage and keep your virtual environment and your credentials updated.

You can **exit the development environment** by accessing the Command Palette (`⇧Shift` + `⌘Cmd` + `P`) and selecting **Dev Containers: Reopen Folder Locally**.

## Test-Driven Development

VSCode's testing capabilities are already set up via the `.vscode/settings.json`. You can access the testing window from VSCode's left sidebar menu
to execute and/or debug the test suite.

We are using `pytest` as our testing framework. You can execute the test suite by running the `pytest` command from the project root directory.
You can also specify the test directory or subdirectory, e.g., `pytest tests/shared`.

## Development Helpers

### Linter and Formatter

We use [ruff](https://docs.astral.sh/ruff/) as our Python code linter and formatter.

Before committing your python code:

- **Lint** it using:

    ```bash
    ruff check --fix src tests lambda_handler.py
    ```

    This will automatically fix minor issues, such as removing unused imports and superfluous whitespaces.

- **Format** it using

    ```bash
    ruff format src tests lambda_handler.py
    ```

### Static Type Checkers

We use [mypy](https://mypy-lang.org/) as our Python static type checker. Based on type hinting, mypy identifies potential bugs related to conflicting types.

Before committing your python code:
- **Type-check** it using:

    ```bash
    mypy src tests lambda_handler.py
    ```

    Fix all flagged issues (if any).

**Note**: mypy is notoriously slow. Its type-checking process may take up to a minute, so please be patient while it runs.

## Available Agents

### Summarize Document Agent

Generates concise summaries with key points extraction.

**Input:**
- `document_text` (str): The document to summarize
- `max_length` (int, optional): Maximum summary length in words (default: 200)
- `language` (str, optional): Output language (default: "English")

**Output:**
- `summary` (str): Concise document summary
- `key_points` (list[str]): 3-5 key points from the document
- `word_count` (int): Approximate word count of the summary

### Extract Entities Agent

Extracts named entities from documents.

**Input:**
- `document_text` (str): The document to analyze
- `entity_types` (list[str], optional): Types of entities to extract

**Output:**
- `people` (list[str]): Names of people mentioned
- `organizations` (list[str]): Organizations mentioned
- `locations` (list[str]): Locations mentioned
- `dates` (list[str]): Dates mentioned
- `key_terms` (list[str]): Important domain-specific terms

### Classify Document Agent

Classifies documents by category, type, and sentiment.

**Input:**
- `document_text` (str): The document to classify
- `custom_categories` (list[str], optional): Custom categories to prioritize

**Output:**
- `primary_category` (str): Primary category of the document
- `secondary_categories` (list[str]): Additional relevant categories
- `confidence_score` (float): Confidence score between 0 and 1
- `sentiment` (str): Overall sentiment (positive, negative, neutral, mixed)
- `document_type` (str): Type of document (article, report, email, etc.)

## Testing Example

```bash
curl --location 'http://127.0.0.1:3005/summarize_document' \
--header 'Content-Type: application/json' \
--data '{
    "document_text": "Your document text here...",
    "max_length": 150,
    "language": "English"
}'
```

## Troubleshooting (Common Issues)

### Local or external dependency is not found

If you get module not found errors from the Python interpreter, such as:

```
ModuleNotFoundError: No module named src.my_local_package
```

or commands not found errors in Bash like

```
ruff: command not found
```

Fix it using the `poetry install` command and re-opening a new terminal window.

### Secret not found

If you get an error message mentioning that the secret was not found from AWS Secret Manager,
ensure that you properly configured your AWS profile. You can find your AWS authentication profile at `~/.aws/credentials`.
Ensure that your `developer-353132507445` profile is set. Also ensure that you are logged in, by hitting `aws sso login --profile developer-353132507445`.

## Using VS Code Interpreter

To play files in VS Code you want to switch VS Code's python interpreter to use poetry.

1. Open the Command Palette: Press `Cmd+Shift+P`.
2. Type `Python: Select Interpreter` and press Enter.
3. Choose the Poetry Environment: A list of available Python interpreters will appear. Look for the one that has your project's name in the path and is labeled with 'Poetry'. Select it.
