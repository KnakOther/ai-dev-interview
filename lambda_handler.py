import sys
import zipfile

# Install dependency zip package to /tmp/ and add to path. It contains external dependencies (imports) not included in the lambda image.
with zipfile.ZipFile("./python_dependencies.zip", "r") as zip_ref:
    zip_ref.extractall("/tmp/")
sys.path.insert(1, "/tmp/")


from src.handlers import (
    classify_document_handler,
    extract_entities_handler,
    summarize_document_handler,
)


def summarize_document_lambda_handler(event, context):
    return summarize_document_handler.handle(event, context)


def extract_entities_lambda_handler(event, context):
    return extract_entities_handler.handle(event, context)


def classify_document_lambda_handler(event, context):
    return classify_document_handler.handle(event, context)
