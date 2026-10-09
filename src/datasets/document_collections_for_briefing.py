from src.shared.dataset import Dataset


class DocumentCollectionsForBriefingDataset(Dataset):
    def __init__(self) -> None:
        self.name = "Document collections for briefing"
        self.description = (
            "Collections of related documents for testing the briefing deep agent."
        )
        self.entries = []

    def load_entries(self):
        """Loads collections of documents for briefing into Langsmith.

        The file must contain a list of dictionaries, where each dictionary has an 'input' field with the
        agent input: a 'documents' list of `{"title", "text"}` dictionaries and an optional 'objective'.
        The bucket name is passed as the `AI_EVALUATION_DATASETS_BUCKET_NAME` environment variable.
        """
        self._load_entries_from_s3_bucket_json("document_collections_for_briefing.json")


if __name__ == "__main__":
    DocumentCollectionsForBriefingDataset().create_and_seed()
