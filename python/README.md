Base64-encoded OpenAI embeddings **in Python**.

Uses [Requests](https://requests.readthedocs.io/en/latest/).

The [OpenAI Python library](https://github.com/openai/openai-python) is also
used, for comparison. In particular, output of `get_embedding` is compared to
Java output converted to `doubles`, in `../compare/check.rb`.
