# Copyright (c) 2023 Eliah Kagan
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

"""
Shared code to read the OpenAI API key.

Some duplication across the two notebooks is okay, so that it is always clear
what's going on, but the logic to get the API key is not interesting.
"""

__all__ = ['get_api_key']

import os
import pathlib


def get_api_key() -> str:
    """
    Get the user's OpenAI API key.

    Two places are searched, in this order:

      1. The content of the ``$OPENAI_API_KEY`` environment variable.

      2. The ``.api_key`` file in the repository root.

    Note that the API key must NOT be committed to this repository! However, if
    the filename ``.api_key`` is excluded in ``.gitignore`` to keep it from
    being committed, then that's okay, at least in development scenarios.
    """
    if api_key := os.getenv('OPENAI_API_KEY', default='').strip():
        return api_key

    pathname = pathlib.Path(__file__).parent.parent / '.api_key'

    with open(pathname, encoding='utf-8') as file:
        return file.read().strip()
