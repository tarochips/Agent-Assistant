class AppError(Exception):
    """Expected business error that can be returned safely to API clients."""

    def __init__(self, message: str, *, code: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class NotFoundError(AppError):
    def __init__(self, message: str, *, code: str = "NOT_FOUND") -> None:
        super().__init__(message, code=code, status_code=404)


class InvalidDocumentError(AppError):
    def __init__(self, message: str, *, code: str = "INVALID_DOCUMENT") -> None:
        super().__init__(message, code=code, status_code=400)
