from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """You are an expert document classification system.

Your task is to analyze documents and classify them according to:
1. Primary category (the main topic or domain)
2. Secondary categories (additional relevant topics)
3. Confidence score (how confident you are in the classification)
4. Sentiment (the overall tone)
5. Document type (the format/purpose)

Common categories include: Business, Technology, Science, Health, Finance, Legal, Marketing, HR, Operations, Product, Engineering, etc.

{custom_categories_instruction}

Guidelines:
- Choose the most specific and accurate primary category
- Include 0-3 secondary categories if relevant
- Provide an honest confidence score based on clarity and content
- Sentiment should be: positive, negative, neutral, or mixed
- Document type should reflect the format: article, report, email, memo, proposal, contract, specification, etc.""",
        ),
        (
            "human",
            "Please classify the following document:\n\n{document_text}",
        ),
    ]
)


def custom_categories_instruction(custom_categories: list[str]) -> str:
    if not custom_categories:
        return ""

    return f"Please prioritize these custom categories if relevant: {', '.join(custom_categories)}"
