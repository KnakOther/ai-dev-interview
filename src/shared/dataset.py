import json
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator

import boto3
from langsmith import Client as LangsmithClient
from langsmith.schemas import Example

from src import DEFAULT_PROFILE_NAME, DEFAULT_REGION_NAME
from src.shared.utils import (
    download_directory_from_s3,
    get_langsmith_api_key,
    upload_directory_to_s3,
)

LANGSMITH_BULK_IMPORT_BYTE_SIZE_LIMIT = 20971520


class Dataset(ABC):
    """Abstract base class for evaluation datasets.

    This class defines the structure for creating and managing evaluation datasets that can be uploaded and
    to Langsmith. Each dataset has a name, description, and a collection of entries
    that must be loaded by implementing the `load_entries` method in subclasses.

    The `create_and_seed` method creates the dataset in Langsmith and seeds it with loaded entries.
    The `fetch_entries` method allows fetching existing entries from the Langsmith service.
    """

    name: str
    description: str
    entries: list[dict]
    langsmith_api_key: str | None = None

    @abstractmethod
    def load_entries(self):
        """Loads the entries from the dataset into the entries attribute.

        Implement this method in the child class to load the entries from the dataset.
        The entries must be a list of DatasetEntries.
        Once the `load_entries` method is implemented, the `Dataset` class can be used to create and seed the dataset.
        """

    def create_and_seed(self) -> None:
        self.load_entries()

        if not self.langsmith_api_key:
            self.langsmith_api_key = get_langsmith_api_key()

        client = LangsmithClient(api_key=self.langsmith_api_key)

        client.create_dataset(
            dataset_name=self.name,
            description=self.description,
        )

        self._create_examples_in_batches(client)

    def fetch_entries(self) -> Iterator[Example]:
        if not self.langsmith_api_key:
            self.langsmith_api_key = get_langsmith_api_key()

        client = LangsmithClient(api_key=self.langsmith_api_key)

        return client.list_examples(dataset_name=self.name)

    def _create_examples_in_batches(self, client: LangsmithClient) -> None:
        batches = self._split_entries_into_batches()
        for batch in batches:
            client.create_examples(
                examples=[{"inputs": entry} for entry in batch],
                dataset_name=self.name,
            )

    def _split_entries_into_batches(self) -> list[list[dict]]:
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_batch_size: int = 0

        for entry in self.entries:
            entry_size = len(
                json.dumps(entry).encode("utf-8")
            )  # Calculate the size of the entry in bytes

            if current_batch_size + entry_size > LANGSMITH_BULK_IMPORT_BYTE_SIZE_LIMIT:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

            current_batch.append(entry)
            current_batch_size += entry_size

        if current_batch:
            batches.append(current_batch)

        return batches

    def _load_entries_from_s3_bucket_json(
        self,
        json_file_name: str,
        region_name: str = DEFAULT_REGION_NAME,
        profile_name: str = DEFAULT_PROFILE_NAME,
    ):
        bucket_name = get_bucket_name()

        session = boto3.session.Session(
            region_name=region_name, profile_name=profile_name
        )
        s3_client = session.client("s3")

        try:
            file_obj = s3_client.get_object(Bucket=bucket_name, Key=json_file_name)
            file_content = file_obj["Body"].read().decode("utf-8")
            self.entries = json.loads(file_content)

        except s3_client.exceptions.NoSuchBucket:
            raise ValueError(f"""The bucket '{bucket_name}' does not exist. Please check that you properly set the
                             AI_EVALUATION_DATASETS_BUCKET_NAME environment variable and that you are 
                             authenticated to the correct AWS Account.""")

        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"""The file '{json_file_name}' was not found in the bucket '{bucket_name}'.
                                    Please check that you uploaded the file to the correct location. and that you 
                                    are authenticated to the correct AWS Account.""")


def get_bucket_name() -> str:
    bucket_name = os.getenv("AI_EVALUATION_DATASETS_BUCKET_NAME")

    if not bucket_name:
        raise ValueError(
            "The environment variable 'AI_EVALUATION_DATASETS_BUCKET_NAME' is not set."
        )

    return bucket_name


def load_file_to_evaluation_bucket(
    local_file_path: str,
    s3_key: str | None = None,
    region_name: str = DEFAULT_REGION_NAME,
    profile_name: str = DEFAULT_PROFILE_NAME,
):
    session = boto3.session.Session(region_name=region_name, profile_name=profile_name)
    s3_client = session.client("s3")
    s3_client.upload_file(
        Filename=local_file_path,
        Bucket=get_bucket_name(),
        Key=s3_key if s3_key else local_file_path,
    )


def download_file_from_evaluation_bucket(
    s3_key: str,
    local_file_path: str | None = None,
    region_name: str = DEFAULT_REGION_NAME,
    profile_name: str = DEFAULT_PROFILE_NAME,
):
    session = boto3.session.Session(region_name=region_name, profile_name=profile_name)
    s3_client = session.client("s3")
    s3_client.download_file(
        Bucket=get_bucket_name(),
        Key=s3_key,
        Filename=local_file_path if local_file_path else s3_key,
    )


def delete_file_in_evaluation_bucket(
    s3_key: str,
    region_name: str = DEFAULT_REGION_NAME,
    profile_name: str = DEFAULT_PROFILE_NAME,
):
    session = boto3.session.Session(region_name=region_name, profile_name=profile_name)
    s3_client = session.client("s3")
    s3_client.delete_object(
        Bucket=get_bucket_name(),
        Key=s3_key,
    )


def load_directory_to_evaluation_bucket(
    local_directory: str, s3_directory: str | None = None
):
    upload_directory_to_s3(
        s3_client=boto3.client("s3"),
        local_directory=local_directory,
        bucket_name=get_bucket_name(),
        s3_directory=s3_directory if s3_directory else local_directory,
    )


def download_directory_from_evaluation_bucket(
    s3_directory: str, local_directory: str | None = None
):
    download_directory_from_s3(
        s3_client=boto3.client("s3"),
        bucket_name=get_bucket_name(),
        s3_directory=s3_directory,
        local_directory=local_directory if local_directory else s3_directory,
    )


def list_files_in_evaluation_bucket(
    prefix: str = "",
    region_name: str = DEFAULT_REGION_NAME,
    profile_name: str = DEFAULT_PROFILE_NAME,
) -> list[str]:
    session = boto3.session.Session(region_name=region_name, profile_name=profile_name)
    s3_client = session.client("s3")
    resp = s3_client.list_objects_v2(Bucket=get_bucket_name(), Prefix=prefix)
    return [obj["Key"] for obj in resp.get("Contents", [])]


if __name__ == "__main__":
    import sys

    cmd, *args = sys.argv[1:]
    if cmd == "upload":
        load_file_to_evaluation_bucket(args[0], args[1] if len(args) > 1 else None)
    elif cmd == "download":
        download_file_from_evaluation_bucket(
            args[0], args[1] if len(args) > 1 else None
        )
    elif cmd == "delete":
        delete_file_in_evaluation_bucket(args[0])
    elif cmd == "bucket":
        print(get_bucket_name())
    elif cmd == "list":
        prefix = args[0] if args else ""
        for k in list_files_in_evaluation_bucket(prefix):
            print(k)
    else:
        print(
            "usage: python -m src.shared.dataset [upload|download|delete|bucket|list] ...",
            file=sys.stderr,
        )
