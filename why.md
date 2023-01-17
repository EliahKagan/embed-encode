<!-- SPDX-License-Identifier: 0BSD -->

# Why embeddings via the Python library show more digits

You probably aren't benefiting from the extra digits, whose presence or absence
mostly reflect different ways of converting between data types. But the Python
library's behavior can be achieved in other languages, if you want it.

In short, the [OpenAI Python library](https://github.com/openai/openai-python)
retrieves embeddings from the API as encoded binary data, rather than using
intermediate decimal representations as is usually done. When these values are
the converted to Python's `float` type, which has higher precision, you get
trailing nonzero digits. Details follow.

## Why you probably don't need the extra digits

The coordinates in most text embeddings, including all embeddings from OpenAI
embedding models, are
[float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
(single precision). So they have 6 to 9 significant figures, though [rarely
9](https://stackoverflow.com/questions/60790120/which-single-precision-floating-point-numbers-need-9-significant-decimal-digits).

When using the OpenAI embeddings API endpoint explicitly via its [documented
interface](https://beta.openai.com/docs/api-reference/embeddings?lang=curl), as
well as when accessing it through *most*
[libraries](https://beta.openai.com/docs/libraries/python-bindings), each
embedding is encoded as a [JSON](https://en.wikipedia.org/wiki/JSON) array of
numbers. Although the embeddings are originally in
[binary](https://en.wikipedia.org/wiki/Binary_number), and they will be in
binary again when we use them, they are, in effect, transmitted in
[decimal](https://en.wikipedia.org/wiki/Decimal).

I *believe* they are always represented with sufficient precision in this
process. Since they are floating point values, we [count from the first nonzero
digit](https://en.wikipedia.org/wiki/Significand) to determine the precision of
a representation. Counting this way, most coordinates from the API endpoint are
transmitted with 8 figures. Some have fewer, but that's probably just when the
eighth digit would be a 0. [Less than 2%](https://stackoverflow.com/a/60790121)
of floating point values (in the whole range of float32) need 9 digits to
support exact round-trip conversion. My guess is that, if one looks for numbers
with 9-digit significands in the JSON results, one may find them.

Of equal or greater importance is that embeddings probably don't need full
float32 precision:

- Embeddings are inherently fuzzy.

  Two texts that are nearly identical strings, in such a way that they have
  same meaning, will usually have at least slightly different embeddings. When
  the difference is small, this is not a problem. But that difference, even
  when it is small, and even when it is due to completely irrelevant changes in
  the text, is often enormously bigger than the distance between adjacent
  float32 values.

- Machine learning models, even in the absence of deliberate or desired
  randomness, are often
  [nondeterministic](https://pytorch.org/docs/stable/notes/randomness.html).

  I'm not sure about the other OpenAI embedding models. But I've observed that
  [text-embedding-ada-002](https://beta.openai.com/docs/guides/embeddings/second-generation-models)—which
  is the [most
  important](https://openai.com/blog/new-and-improved-embedding-model/) of them
  as of this writing—will occasionally return different embeddings of the same
  text, even when the embedding is requested in exactly the same way. (I wish I
  had a good citation for this, which is interesting in its own right.)

## What the extra digits usually represent

### An illustrative thought experiment

Suppose we receive a decimal representation of a
[float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
(single precision) value, but we will ultimately be calculating in
[float64](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)
(double precision). Suppose further that the float32 value is represented in
decimal with enough digits that, when parsed back to float32, the original and
round-tripped values are exactly equal. If the decimal representation is
instead parsed as float64, will that also be equal to the original number?

Surprisingly, *no*. A float32 value round-trips perfectly when no other float32
value is closer to the decimal value used to represent it. But there will
usually be a closer float64 value. That float64 value is a more exact
representation *of the intermediate decimal value*.

If the decimal value had enough precision to uniquely represent the float32
value, then the float64 value obtained directly from the decimal value is
*slightly farther away* from the original float32 value than if it had been
parsed as float32 and *then* converted to float64. Thus, while it is rare that
this would matter in practice, the rounding error may be *slightly* decreased
by converting values that originated as float32 back to float32 before
converting them to float64.

The reason it is rare that it would matter is that we are talking about
differences less than the precision of float32. But if that is likely to
matter, then we need to fix the problem of using float32 in the first place,
instead!

If we follow that two-step process and then represent those float64 values in
decimal, we see extra digits. The extra digits correspond to the difference
between the decimal representations of the float32 values and the original
float32 values themselves. Note that this assumes perfect round-tripping;
otherwise, we have no way to know the original float32 values exactly.

### Trying it out in the Python REPL

An example may be illuminating. Python has no built-in float32 type; its
`float` type is, in practice, float64 on all architectures. But
[NumPy](https://numpy.org/) does have float32. Suppose we have a float32 value
represented as 0.033652876. This is a round-trip representation:

```python
>>> import numpy as np
>>> np.float32('0.033652876')
0.033652876
```

Like all round-trip representations of a float32, it is also a round-trip
representation of a float64 (though, as explained above, this is a slightly
different float64):

```python
>>> float('0.033652876')
0.033652876
>>> float('0.033652876') - np.float32('0.033652876')
4.2755698981267187e-10
```

Note that the difference there is very small.

Converting the decimal representation to float32 and converting *that* to
float64 gives:

```python
>>> float(np.float32('0.033652876'))
0.03365287557244301
>>> float(np.float32('0.033652876')) - np.float32('0.033652876')
0.0
```

Notice the extra digits.

## What the OpenAI Python library does

The reason you get extra digits when you view representations of the Python
`float`s obtained using the [OpenAI Python
library](https://github.com/openai/openai-python) is similar to, but **not
quite the same as**, the above scenario. The difference is that the library
processes the embeddings in float32 not due to *parsing* them as float32, but
because it uses the API endpoint in a way that causes it to receive float32
values in the first place—the values are not represented in decimal during
transmission from the API endpoint. It achieves this by passing `base64` as the
value of the optional `encoding_format` argument.

**Beware:** This is done as an optimization. The `encoding_format` argument is
not
[documented](https://github.com/openai/openai-openapi/blob/master/openapi.yaml).
As far as I know, it could be removed or changed at any time in the future. The
OpenAI Python library is written in a way that assumes is not an error to pass
this argument to the API endpoint, but it does *not* assume that the argument
is heeded. If the API endpoint simply ignores the argument, the OpenAI Python
library gracefully accepts the default representation (a JSON list of numbers).
Therefore, OpenAI could, at some point in the future, drop support for
`encoding_format` from the API endpoint, even without (or before) releasing a
patch to the Python library.

### How the OpenAI Python library retrieves embeddings

There are two ways to use the OpenAI Python library to get embeddings:

- Call
  [`openai.Embedding.create`](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/embedding.py#L14).

- Use the
  [`openai.embeddings_utils`](https://github.com/openai/openai-python/blob/v0.26.1/openai/embeddings_utils.py)
  module which offers
  [`get_embedding`](https://github.com/openai/openai-python/blob/v0.26.1/openai/embeddings_utils.py#L17)
  and
  [`get_embeddings`](https://github.com/openai/openai-python/blob/v0.26.1/openai/embeddings_utils.py#L39)
  functions. These offer a higher level interface to `openai.Embedding.create`.

(There are also [asynchronous
versions](https://github.com/openai/openai-python/pull/146) of all those
functions, which I am omitting for simplicity.)

Either way, `openai.Embedding.create` is ultimately used. The `get_embedding`
and `get_embeddings` functions call `openai.Embedding.create`. They do not pass
an `encoding_format` argument.

When `openai.Embedding.create` is called without an `encoding_format`
argument—whether because you used those higher level functions, or because you
called it directly and used only officially documented arguments—it requests
the float32 coordinates as a [Base64](https://en.wikipedia.org/wiki/Base64)
string. Base64 encodes binary data losslessly in a text format suitable for
transmission over a network. Note that the "64" in base64 is not conceptually
important here. In particular, it is completely unrelated to the distinction
between float32 and float64.

`openai.Embedding.create` decodes the Base64 string to get the original binary
representations of the coordinates, then uses [NumPy](https://numpy.org/) to
recognize that as a [flat
sequence](https://www.fluentpython.com/lingo/#flat_sequence) of float32 values,
making a NumPy array of them. Then it converts that NumPy array to a Python
`list`, which converts the float32 values to Python `float`. As [mentioned
above](#Trying-it-out-in-the-Python-REPL), Python's `float` type is in practice
always float64.

### The actual code that does this

That program logic appears in [this part of the
code](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/embedding.py#L27)
from the
[`openai.Embedding.create`](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/embedding.py#L14)
method, in
[`api_resources/embedding.py`](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/embedding.py):

```python
# If encoding format was not explicitly specified, we opaquely use base64 for performance
if not user_provided_encoding_format:
    kwargs["encoding_format"] = "base64"

while True:
    try:
        response = super().create(*args, **kwargs)

        # If a user specifies base64, we'll just return the encoded string.
        # This is only for the default case.
        if not user_provided_encoding_format:
            for data in response.data:

                # If an engine isn't using this optimization, don't do anything
                if type(data["embedding"]) == str:
                    assert_has_numpy()
                    data["embedding"] = np.frombuffer(
                        base64.b64decode(data["embedding"]), dtype="float32"
                    ).tolist()

        return
```

(I've omitted the surrounding code, including the `except` clause.)

### The fallback logic (if the API endpoint doesn't give Base64)

That code seems to indicate that not all models are guaranteed to support
Base64 encoding. I don't know if this is because any OpenAI embedding model
currently does not, or if it is only to avoid committing OpenAI to continuing
to offer `encoding_format` in the future.

If the model doesn't support Base64 encoding, then the API endpoint would fall
back to giving the `embedding` value as a JSON array. Then, due to the parsing
logic in the
[`EngineAPIResource`](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/abstract/engine_api_resource.py#L15)
base class,
[`type(data["embedding"])`](https://github.com/openai/openai-python/blob/v0.26.1/openai/api_resources/embedding.py#L41)
would be `list` rather than `str`.

Using [Requests](https://requests.readthedocs.io/en/latest/), I tested with
[text-embedding-ada-002](https://beta.openai.com/docs/guides/embeddings/second-generation-models),
as well as the five first-generation Ada models (see [Embedding
models](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings#first-generation-models)).
In my tests, `'encoding_format': 'base64'` always worked to get Base64-encoded
embeddings. Those tests are in
[`python/several-models.ipynb`](python/several-models.ipynb) in this
repository.

## To receive and decode Base64 yourself...

The float64 values you get from the OpenAI Python library are being produced
because Python has no built-in float32 type, not because the extra precision
would usually be helpful. But if you like, you can use the same technique
`openai.Embedding.create` uses: when accessing the API endpoint directly,
specify `'encoding_format': 'base64'` and decode the results yourself. **Note
that this does not appear to be officially supported, so your code could break
at any time.**

This repository shows ways to retrieve Base64-encoded embeddings by explicitly
requesting them from the API endpoint, in three languages:

- [In Bash](bash/README.md), using [`curl`](https://curl.se/docs/manpage.html),
  [`jq`](https://stedolan.github.io/jq/manual/), and
  [`base64`](https://linux.die.net/man/1/base64). See the shell scripts
  [`demo`](bash/demo) and [`demo-short`](bash/demo-short).

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

There is also [a comparison](compare/README.md) of the results of OpenAI Python
library and the Java code in this repository, after conversion to float64.
