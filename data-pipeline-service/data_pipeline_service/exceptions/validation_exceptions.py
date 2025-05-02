class ValidationError(Exception):
    """
    Exception raised for validation errors in data input or processing.
    
    Attributes:
        message -- explanation of the error
    """
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)