<!-- SPDX-License-Identifier: 0BSD -->

# Why embeddings via the Python library show more digits

You probably aren't benefiting from the extra digits, whose presence or absence
mostly reflect different ways of converting between data types. But the Python
library's behavior can be achieved in other languages, if you want it.

## Why you probably don't need the extra digits

The coordinates in most text embeddings, including all embeddings from OpenAI
embedding models, are
[float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
(single precision). So their coordinates have 6 to 9 significant figures,
though [rarely
9](https://stackoverflow.com/questions/60790120/which-single-precision-floating-point-numbers-need-9-significant-decimal-digits).

When using the OpenAI embeddings API endpoint explicitly via its [documented
interface](https://beta.openai.com/docs/api-reference/embeddings?lang=curl), as
well as when accessing it through *most*
[libraries](https://beta.openai.com/docs/libraries/python-bindings), each
embedding is encoded as a [JSON](https://en.wikipedia.org/wiki/JSON) array of
numbers. Although the embeddings are originally in binary, and they will be in
binary again when we use them, they are, in effect, transmitted in decimal.

I *believe* they are always represented with sufficient precision in this
process. Since they are floating point values, we [count from the first nonzero
digit](https://en.wikipedia.org/wiki/Significand) to determine the precision of
a representation. Counting this way, most coordinates from the API endpoint are
transmitted with 8 figures. Some have fewer, but that's probably just when the
eighth digit would be a 0. Less than 2% of floating point values (in the whole
range of float32) need 9 digits to support exact round-trip conversion. My
guess is that, if one looks for such numbers in the JSON results, one may find
them.

Of at least equal importance is that embeddings probably don't need full
float32 precision:

- Embeddings are inherently fuzzy.

  Two texts that are nearly identical strings, in such a way that they have
  same meaning, will usually have at least slightly different embeddings. When
  the difference is small, this is not a problem. But that difference, even
  when it is small, and even when it is due to completely irrelevant changes in
  the text, is often a lot bigger than the distance between adjacent float32
  values.

- Machine learning models, even in the absence of deliberate or desired
  randomness, are often
  [nondeterministic](https://pytorch.org/docs/stable/notes/randomness.html).

  I'm not sure about the other OpenAI embedding models, but I've observed that
  [text-embedding-ada-002](https://beta.openai.com/docs/guides/embeddings/second-generation-models),
  the [most important of them as of this
  writing](https://openai.com/blog/new-and-improved-embedding-model/), will
  occasionally return different embeddings of the same text, even when the
  embedding is requested in exactly the same way. (I wish I had a good citation
  for this, which is interesting in its own right.)

## What the extra digits represent

Suppose we receive a decimal representations of a
[float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
(single precision) value, but we will ultimately be calculating with it in
[float64](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)
(double precision). Suppose further that the float32 value is represented in
decimal, with enough digits that, when parsed back to float32, the original and
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
this would matter in practice, the rounding error is may be *slightly*
decreased by converting values that originated a float32 back to float32 before
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

An example may be illustrative. Python has no built-in float32 type; its
`float` type is, in practice, float64 on all architectures. But
[NumPy](https://pypi.org/project/numpy/) does have float32. Suppose we have a
float32 value represented as 0.033652876. This is a round-trip representation:

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

Converting the decimal representation to float32 and converting *that* to float64 gives:

```python
>>> float(np.float32('0.033652876'))
0.03365287557244301
>>> float(np.float32('0.033652876')) - np.float32('0.033652876')
0.0
```

Notice the extra digits.

## What the OpenAI Python library does

The reason you get extra digits in the decimal representations of `float`s
obtained using the OpenAI Python library is similar to, but **not quite the
same as**, the above scenario. The difference is that the library processes the
embeddings in float32 not due to *parsing* them as float32, but because it uses
the API in a special way that causes it to have float32 values in the first
place--the values are not represented in decimal during transmission from the
API endpoint.

---

Nonetheless, it remains interesting that the [OpenAI Python
library](https://github.com/openai/openai-python) gives results that, when
printed, show more decimal digits, even compared to accessing the API endpoint
explicitly in Python using
[Requests](https://requests.readthedocs.io/en/latest/). This behavior is worth
explaining--even though it could change in the future. It turns out to relate
to an interesting optimization that the API endpoint facilitates and the OpenAI
Python library takes advantage of: the `encoding_format` argument.

***Unrevised (the revised material is above the immediately preceding `hr`):***

More specifically, you can get the embeddings as flat sequences of binary
float32 values--which is what the Python library is doing--and use them however
you like.

There are two ways to use the OpenAI Python library to get embeddings. One is
to call `openai.Embedding.create`. The other is to use the
`openai.embeddings_utils` module which offers `get_embedding` and
`get_embeddings` functions, which offer a higher level interface to
`openai.Embedding.create`. (There are also asynchronous functions, which I am
omitting for simplicity.)

Both ways ultimately work by using the OpenAI embeddings API endpoint. The API endpoint, at least currently, supports an `encoding_format` argument, which [is not officially documented]() as of this writing.

If I understood your message in #text-embedding-ada-002 correctly, you're using `openai.embeddings_utils.get_embedding`. That calls `openai.Embedding.create`, and does not pass an `encoding_format` argument. When `openai.Embedding.create` is called without an `encoding_format` argument, it requests the float32 coordinates as a base64 string. base64 encodes binary data losslessly in a text format suitable for transmission over a network. (The "64" in base64 is not conceptually important here.)

`openai.Embedding.create` decodes the base64 string to get the original binary representations of the coordinates, then uses NumPy to recognize that as a flat sequence of float32 values, making a NumPy array of them. Then it converts that NumPy array to a Python `list`, which converts the float32 values to Python `float`. Python's `float` type is in practice always float64 (double precision). Python itself has no built-in float32 type.

That program logic appears in this code from the `openai.Embedding.create` method, in https://github.com/openai/openai-python/blob/main/openai/api_resources/embedding.py:
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

        return response
```
(I've omitted the surrounding code, including the `except` clause.)

Since float32 and float64 are *binary* floating point (their IEEE 754 names are binary32 and binary64), there is error associated with the base conversion between decimal and binary. For example, in a REPL, `np.float32('0.033652876')` and `float('0.033652876')` both show the string representation `0.033652876`, while `float(np.float32('0.033652876'))` shows the string representation `0.03365287557244301`.

The float64 values you get from the OpenAI Python library are being produced because Python has no built-in float32 type, not because the extra precision would usually be helpful. But if you like, you can use the same technique `openai.Embedding.create` uses: when accessing the API endpoint directly, specify `'encoding_format': 'base64'` and decode the results yourself.

The only thing that makes me a little uneasy is that I haven't found where `encoding_format` is officially documented. The code I showed above seems to indicate that not all models are guaranteed to support base64 encoding. If the model doesn't, then presumably the API endpoint would fall back to giving the `embedding` value as a JSON array and, due to the parsing logic in the `EngineAPIResource` base class, `type(data["embedding"])` would be `list` rather than `str`. Using `requests`, I tested with `text-embedding-ada-002`, as well as the five first-generation Ada models, and `'encoding_format': 'base64'` always worked to get base64-encoded embeddings.
