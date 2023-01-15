<!-- SPDX-License-Identifier: 0BSD -->

The embeddings are float32 (single precision). They only have 7 or 8 significant figures, so you probably aren't benefiting from the extra digits, whose presence or absence mostly reflect different ways of converting between data types. But you *can* get them if you want, even when you're not using the OpenAI Python library. More specifically, you can get the embeddings as flat sequences of binary float32 values--which is what the Python library is doing--and use them however you like.

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
