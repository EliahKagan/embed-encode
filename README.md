<!-- SPDX-License-Identifier: 0BSD -->

# Base64-encoded embeddings from the OpenAI API

Displaying coordinates of text embeddings retrieved using the OpenAI Python
library shows more digits than when the embeddings are retrieved explicitly
from the API endpoint or using most other libraries. This repository explores
why that is, how to get this behavior (and by the same mechanism) when working
in other languages, and why one should not usually bother to do so.

More specifically, this repository is a collection of code examples and
documentation for the `encoding_format` argument to the OpenAI embeddings API,
which, when set to `base64`, will send raw floats encoded in Base64. The OpenAI
Python library uses that under the hood.

Beware that `encoding_format` is not officially documented. It could be removed
or changed in the future!

## License

This project is licensed under [0BSD](https://spdx.org/licenses/0BSD.html),
which is a ["public-domain
equivalent"](https://en.wikipedia.org/wiki/Public-domain-equivalent_license)
license. See
[**`LICENSE`**](https://gist.github.com/EliahKagan/97e4b60c5c77f062c41e34bd42ec75f8#file-license)
for details.

## Acknowledgements

These materials arose out of conversations with
[**RonaldGRuckus**](https://discordapp.com/users/911807261725294602/) on the
OpenAI Discord server. If not for Ronald's observations about embeddings from
the Python library, and the conversations that followed, this repository and
its contents would not exist.

## Contents

See **[Why embeddings via the Python library show more digits](why.md)** for a
fully detailed explanation of this.

The example code in this repository is in three directories:

- [In Bash](shell/README.md), using
  [`curl`](https://curl.se/docs/manpage.html),
  [`jq`](https://stedolan.github.io/jq/manual/), and
  [`base64`](https://ss64.com/bash/base64.html). See the shell scripts
  [`demo`](shell/demo) and [`demo-short`](shell/demo-short).

- [In Python](python/README.md), using
  [Requests](https://requests.readthedocs.io/en/latest/). See the notebooks
  [`ada-002.ipynb`](python/ada-002.ipynb) and
  [`several-models.ipynb`](python/ada-002.ipynb).

- [In Java](java/README.md), using [OkHttp](https://square.github.io/okhttp/)
  and [Jackson](https://github.com/FasterXML/jackson). See
  [`Embedder.java`](java/src/main/java/io/github/eliahkagan/embed_encode/Embedder.java)
  (and
  [`Main.java`](java/src/main/java/io/github/eliahkagan/embed_encode/Main.java)
  for use).

Note that the reason to use `encoding_format`, if there is one, would not
ordinarily be increased precision, but instead the optimization in speed and
network usage, which [appears to be why](why.md#why-it-is-an-optimization) the
OpenAI Python library uses it.

Furthermore, to reiterate the above warning, `encoding_format` is not
officially documented, and it could potentially be removed, or changed, at any
point in the future. The OpenAI Python library's source code [shows
how](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/embedding.py#L40)
one might approach using it in a way that partially avoids depending on its
future existence.
