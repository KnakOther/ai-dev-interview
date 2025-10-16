from src.shared.dataset import Dataset


class DocumentsWithEntitiesDataset(Dataset):
    def __init__(self) -> None:
        self.name = "Documents with entities"
        self.description = "A collection of documents containing named entities for extraction testing."
        self.entries = []

    def load_entries(self):
        """Loads a collection of documents with entities into Langsmith.

        The file must contain a list of dictionaries, where each dictionary has a 'document_text' field
        with the document text containing entities, and optional 'expected_entities' field for validation.
        The bucket name is passed as the `AI_EVALUATION_DATASETS_BUCKET_NAME` environment variable.
        """
        self._load_entries_from_s3_bucket_json("documents_with_entities.json")


if __name__ == "__main__":
    DocumentsWithEntitiesDataset().create_and_seed()
