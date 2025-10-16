from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """You are an expert at extracting named entities from documents.

Your task is to identify and extract the following types of entities:
{entity_types}

Guidelines:
- Only extract entities that are explicitly mentioned in the text
- For people, include full names when available
- For organizations, use the official or most complete name mentioned
- For locations, be specific (city, state, country as mentioned)
- For dates, preserve the format mentioned in the text
- For key terms, identify important technical, domain-specific, or contextually significant terms
- If no entities of a certain type are found, return an empty list for that type""",
        ),
        (
            "human",
            "Please extract entities from the following document:\n\n{document_text}",
        ),
    ]
)
