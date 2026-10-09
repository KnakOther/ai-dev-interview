from src.shared.dataset import Dataset


class DocumentsForSummarizationDataset(Dataset):
    def __init__(self) -> None:
        self.name = "Documents for summarization"
        self.description = (
            "A collection of documents for testing summarization capabilities."
        )
        self.entries = []

    def load_entries(self):
        """Loads a collection of documents for summarization into Langsmith.

        The file must contain a list of dictionaries, where each dictionary has a 'document_text' field
        with the document text, and optional 'metadata' field with additional information.
        The bucket name is passed as the `AI_EVALUATION_DATASETS_BUCKET_NAME` environment variable.
        """
        self._load_entries_from_s3_bucket_json("documents_for_summarization.json")


if __name__ == "__main__":
    DocumentsForSummarizationDataset().create_and_seed()
