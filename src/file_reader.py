def read_file(path: str) -> str:
    """Read a text file and return its content as a string."""
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    text = " ".join(text.split())  # Basic cleanup to remove extra whitespace
    return text