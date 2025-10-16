import os
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import BaseClient

from src import DEFAULT_PROFILE_NAME, DEFAULT_REGION_NAME
from src.shared.constants import GEMINI_FLASH_2_5


def runs_in_debug_mode() -> bool:
    """Detects if the code is running in VSCode debug mode."""
    return sys.monitoring.get_tool(sys.monitoring.DEBUGGER_ID) is not None


def get_langsmith_api_key(
    region_name: str = DEFAULT_REGION_NAME, profile_name: str = DEFAULT_PROFILE_NAME
) -> str:
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        return api_key

    session = boto3.session.Session(region_name=region_name, profile_name=profile_name)
    client = session.client(service_name="secretsmanager")

    try:
        response = client.get_secret_value(SecretId="testing/ai/langsmith/apikey")
        return response["SecretString"]

    except client.exceptions.ResourceNotFoundException:
        raise ValueError("LangsmithApiKey secret not found in AWS Secrets Manager")

    except Exception as e:
        raise ValueError(f"Failed to fetch LangsmithApiKey from Secrets Manager: {e}")


def download_directory_from_s3(
    s3_client: BaseClient,
    bucket_name: str,
    s3_directory: str,
    local_directory: str | Path,
):
    """Downloads all files from a specified directory (prefix) in an S3 bucket to a local directory.

    Parameters:
        s3_client (S3): An instiated boto3 S3 client (e.g. `boto3.client('s3')`).
        bucket_name (str): The name of the S3 bucket.
        s3_directory (str): The S3 directory (prefix) to download. Should end with a '/'.
        local_directory (str): The local directory where files will be downloaded.

    The function preserves the directory structure of the S3 folder in the local directory.
    """
    if (not s3_directory.endswith("/")) and (s3_directory != ""):
        raise ValueError("The S3 directory must end with a '/'.")

    # Create the local directory if it doesn't exist
    os.makedirs(local_directory, exist_ok=True)

    # List objects in the specified S3 directory
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_directory)

    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                # Get the S3 object's key
                s3_key = obj["Key"]

                # Skip if the object is a folder (ends with '/')
                if s3_key.endswith("/"):
                    continue

                # Remove the S3 directory prefix from the key to create the local path
                relative_path = s3_key[len(s3_directory) :].lstrip("/")
                local_file_path = os.path.join(local_directory, relative_path)

                # Ensure the local subdirectories exist
                local_subdir = os.path.dirname(local_file_path)
                if not os.path.exists(local_subdir):
                    os.makedirs(local_subdir)

                # Download the S3 object
                s3_client.download_file(bucket_name, s3_key, local_file_path)


def upload_directory_to_s3(
    s3_client: BaseClient,
    local_directory: str | Path,
    bucket_name: str,
    s3_directory: str,
):
    """Uploads all files from a local directory to a specified directory (prefix) in an S3 bucket.

    Parameters:
        s3_client (S3): An instiated boto3 S3 client (e.g. `boto3.client('s3')`).
        local_directory (str | Path): The local directory to upload.
        bucket_name (str): The name of the S3 bucket.
        s3_directory (str): The S3 directory (prefix) to upload to. Should end with a '/'.

    The function preserves the directory structure when uploading to S3.
    """
    if (not s3_directory.endswith("/")) and (s3_directory != ""):
        raise ValueError("The S3 directory must end with a '/'.")

    local_directory = Path(local_directory)
    if not local_directory.exists():
        raise ValueError(f"The local directory '{local_directory}' does not exist.")

    # Walk through the local directory
    for root, _, files in os.walk(local_directory):
        for filename in files:
            # Get the full local path
            local_file_path = Path(root) / filename

            # Calculate the relative path from the base directory
            relative_path = local_file_path.relative_to(local_directory)

            # Create the S3 key by joining the S3 directory and relative path
            s3_key = s3_directory + str(relative_path)

            # Upload the file
            s3_client.upload_file(
                str(local_file_path),
                bucket_name,
                s3_key,
            )


def setup_gcp_credentials(
    region_name=DEFAULT_REGION_NAME, profile_name: Optional[str] = None
):
    """Sets up Google Cloud credentials in the environment, pulling from the AWS Secrets Manager."""
    if profile_name:
        session = boto3.session.Session(
            region_name=region_name, profile_name=profile_name
        )
    else:
        session = boto3.session.Session(region_name=region_name)

    client = session.client(service_name="secretsmanager", region_name=region_name)
    google_cloud_secret_arn = _get_google_cloud_secret_arn(client)

    google_cloud_secret = client.get_secret_value(SecretId=google_cloud_secret_arn)

    pathdd = "/tmp/gcp_credentials.json"

    with open(pathdd, "w") as f:
        f.write(google_cloud_secret["SecretString"])

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = pathdd


def _get_google_cloud_secret_arn(client: BaseClient) -> str:
    google_secret = os.environ.get("GoogleSecret")

    if not google_secret:
        raise ValueError("The environment variable 'GoogleSecret' is not set.")

    return client.describe_secret(SecretId=google_secret)["ARN"]


def get_file_content_from_s3(
    file_key: str,
    bucket_name: str,
    region_name: str = "us-east-1",
) -> str:
    """Retrieve the content of a file stored in S3.

    When running locally, uses profile_name for authentication.
    When running in Lambda, inherits IAM role permissions.

    Args:
        file_key: S3 key of the file to retrieve
        bucket_name: Name of the S3 bucket
        region_name: AWS region name
        profile_name: AWS profile name (used locally, ignored in Lambda)

    Returns:
        File content as string

    Raises:
        ValueError: If bucket doesn't exist
        FileNotFoundError: If file doesn't exist in bucket
    """

    s3_client = boto3.client("s3", region_name=region_name)

    try:
        file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        return file_obj["Body"].read().decode("utf-8")

    except s3_client.exceptions.NoSuchBucket:
        raise ValueError(f"The bucket '{bucket_name}' does not exist.")

    except s3_client.exceptions.NoSuchKey:
        raise FileNotFoundError(
            f"The file '{file_key}' was not found in the bucket '{bucket_name}'."
        )

    except Exception as e:
        raise ValueError(f"Failed to fetch file from S3: {e}")


def select_model() -> str:
    """Returns the default model for document processing tasks."""
    return GEMINI_FLASH_2_5
