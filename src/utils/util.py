"""Utility functions."""


def convert_to_snake_case(text: str) -> str:
    """Convert text to snake case."""
    return text.lower().replace(" ", "_")
