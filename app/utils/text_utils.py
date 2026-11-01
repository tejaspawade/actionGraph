import re

def normalize_whitespace(text: str) -> str:
    """
    Remove unnecessary spaces and normalize line breaks.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def normalize_speaker_names(text: str) -> str:
    """
    Normalize speaker names appearing before ':'.

    Example:
        "  rahul: Hello"
        "PRIYA : Hi"

    becomes:

        "Rahul: Hello"
        "Priya: Hi"
    """

    lines = []

    for line in text.split("\n"):

        match = re.match(r"^([A-Za-z][A-Za-z0-9 _-]*)\s*:\s*(.*)$", line)

        if match:
            speaker = match.group(1).strip().title()
            message = match.group(2).strip()

            lines.append(f"{speaker}: {message}")
        else:
            lines.append(line)

    return "\n".join(lines)


def clean_transcript(text: str) -> str:
    """
    Apply all transcript cleaning operations.
    """

    text = normalize_whitespace(text)
    text = normalize_speaker_names(text)

    return text