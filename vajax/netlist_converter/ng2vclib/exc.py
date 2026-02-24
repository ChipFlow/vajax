from .generators import format_history


# Define out own exception
class ConverterError(Exception):
    def __init__(self, message, history=None, lineno=None):
        if history is not None:
            txt = format_history(history, lineno) + "\n  " + message
        else:
            txt = message
        super().__init__(txt)
