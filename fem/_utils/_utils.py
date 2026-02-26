

class NonConstantJacobian(Exception):
    """
    Raised when a quadrilateral element is not aligned with the x-axis.
    """
    def __init__(self, element_index, message=None):
        if message is None:
            message = f"Quadrilateral element {element_index} is not aligned with the x-axis."
        super().__init__(message)
        self.element_index = element_index