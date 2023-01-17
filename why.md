<!-- SPDX-License-Identifier: 0BSD -->

# Why embeddings via the Python library show more digits

*You probably aren't benefiting from the extra digits, whose presence or
absence mostly reflect different ways of converting between data types. But the
Python behavior can be achieved in other languages, if you want it.*

The coordinates in most text embeddings are
[float32](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
(single precision). The OpenAI embedding models work this way. So coordinates
in the embeddings they produce have 6 to 9 significant figures, though [rarely
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
trailing digit would be a 0. Less than 2% of floating point values (in the
whole range of float32) need 9 digits to support exact round-trip conversion,
so we shouldn't expect to see that many very often. My guess is that, if one
looks hard for such numbers in the JSON results, one may find them.

Of at least equal importance is that embeddings probably don't need full
float32 precision. Machine learning models, even in the absence of deliberate
or desired randomness, are often
[nondeterministic](https://pytorch.org/docs/stable/notes/randomness.html). I'm
not sure about the other OpenAI embedding models, but `text-embedding-ada-002`
(the most important of them as of this writing) will occasionally return
different embeddings of the same text, even when the embedding is requested in
exactly the same way. (I wish I had a strong citation for this, which is
interesting in its own right, but so far, all I have is personal experience.)

Nonetheless, it remains interesting that the OpenAI Python library gives
results that, when printed, show more decimal digits, even compared to
accessing the API endpoint explicitly in Python using Requests. This behavior
is worth explaining--even though it could change in the future. It turns out to
relate to an interesting optimization that the API endpoint facilitates and the
OpenAI Python library takes advantage of: the `encoding_format` argument.

---

*Unrevised (the revised material is above the immediately preceding `hr`):*

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
