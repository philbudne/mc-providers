"""
minimal stub file
"""

class _FastText:
    def predict(self, text: list[str]) -> list[list[list[str]]]: ...

def load_model(path: str) -> _FastText: ...
