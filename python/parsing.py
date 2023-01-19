"""Parsing information from decimal numbers represented as raw text."""

__all__ = ['group_by_mantissa_length', 'extract_coordinate_strings']

import collections
import re
from typing import Iterable

_EXPECTED_DIMENSION = 1536
"""The correct number of coordinates in a text-embedding-ada-002 embedding."""

_EMBEDDING = re.compile(r'"embedding": \[([^\]]+)\]')
"""Regex for an embedding in raw JSON text."""

_SEPARATOR = re.compile(r'[\s,]+')
"""Regex for a separator in a raw JSON array."""

_NUMBER_CRUFT = re.compile(r'[+-.]|[eE].+')
"""Regex for parts of a number that we always ignore for the mantissa."""

_LEADING_ZEROS = re.compile(r'^0+')
"""Regex for zeros that come before any other character."""


def extract_coordinate_strings(response_text: str) -> list[str]:
    """Get an embedding's coordinates as strings from a raw JSON response."""
    raw_json_array = _extract_raw_coordinate_array(response_text)
    decimals = [token for token in _SEPARATOR.split(raw_json_array) if token]
    _check_dimension(len(decimals))
    return decimals


def _extract_raw_coordinate_array(response_text: str) -> str:
    """Get an embedding JSON array as a string from a raw JSON response."""
    regex_match = _EMBEDDING.search(response_text)
    if not regex_match:
        raise ValueError('JSON array of coordinates not found')
    return regex_match.group(1)


def _check_dimension(dimension: int) -> None:
    """Raise ValueError if a dimension is wrong for text-embedding-ada-002."""
    if dimension != _EXPECTED_DIMENSION:
        message = f'expected dimension {_EXPECTED_DIMENSION}, got {dimension}'
        raise ValueError(message)


def group_by_mantissa_length(decimals: Iterable[str]) -> dict[int, list[str]]:
    """Group text representations of decimal numbers by mantissa length."""
    groups = collections.defaultdict(list)
    for decimal in decimals:
        groups[_mantissa_length(decimal)].append(decimal)
    return dict(groups)


def _mantissa_length(decimal: str) -> int:
    """Given a decimal number as text, count mantissa digits."""
    all_figures = _NUMBER_CRUFT.sub('', decimal)
    mantissa_figures = _LEADING_ZEROS.sub('', all_figures)
    return len(mantissa_figures)
