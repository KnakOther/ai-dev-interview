class FriendlyException(Exception):
    """Exception that is caught and handled gracefully.

    Used to provide a user-friendly error message.
    A FriendlyException is logged as a warning and does not trigger a retry.
    """


class LlmRaisedException(FriendlyException):
    """Exception raised when an LLM agent identifies an error.

    Used in cases where the prompt defines a protocol for the LLM to raise errors.
    """

    def __init__(
        self,
        message: str,
        llm_generated_message: str | None = None,
    ):
        self.message = message
        self.llm_generated_message = llm_generated_message

        super().__init__(self.message)

    def __str__(self):
        _llm_generated_message = (
            f"\n\nThe LLM agent generated the following message: {self.llm_generated_message}"
            if self.llm_generated_message
            else ""
        )
        return self.message + _llm_generated_message
