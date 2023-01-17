"""
Shared code to read the OpenAI API key.

Some duplication across the two notebooks is okay, so that it is always clear
what's going on, but the logic to get the API key is not interesting.
"""

__all__ = ['get_api_key']

import os
import pathlib


def get_api_key():
    """
    Get the user's OpenAI API key.

    Two places are searched, in this order:

      1. The content of the ``$OPENAI_API_KEY`` environment variable.

      2. The ``.api_key`` file in the repository root.

    Note that the API key must NOT be committed to this repository! However, if
    the filename .api_key is excluded in .gitignore to prevent it from being
    committed, then that is acceptable, at least in development scenarios.
    """
    if api_key := os.getenv('OPENAI_API_KEY', default='').strip():
        return api_key

    pathname = pathlib.Path(__file__).parent.parent / '.api_key'

    with open(pathname, encoding='utf-8') as file:
        return file.read().strip()
