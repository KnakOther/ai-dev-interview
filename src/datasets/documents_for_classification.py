from src.shared.dataset import Dataset


class DocumentsForClassificationDataset(Dataset):
    def __init__(self) -> None:
        self.name = "Documents for classification"
        self.description = "A collection of documents for testing classification capabilities."
        self.entries = []

    def load_entries(self):
        """Loads a collection of documents for classification into Langsmith.

        The file must contain a list of dictionaries, where each dictionary has a 'document_text' field
        with the document text, and optional 'expected_category' field for validation.
        The bucket name is passed as the `AI_EVALUATION_DATASETS_BUCKET_NAME` environment variable.
        """
        self._load_entries_from_s3_bucket_json("documents_for_classification.json")


if __name__ == "__main__":
    DocumentsForClassificationDataset().create_and_seed()
