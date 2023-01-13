The embeddings are float32 (single precision). They only have 7 or 8 significant figures, so you probably aren't benefiting from the extra digits. But you *can* get them if you like, even when you're not using the Python library.

`openai.embeddings_utils.get_embedding` calls `openai.Embedding.create`, and does not pass an `encoding_format` argument. When `openai.Embedding.create` is called without an `encoding_format` argument, it requests the float32 coordinates as a base64 string. base64 encodes binary data losslessly in a text format suitable for transmission over a network. (The "64" in base64 is not conceptually important here.)

`openai.Embedding.create` decodes the base64 string to get the original binary representations of the coordinates, then uses NumPy to recognize that as a flat sequence of float32 values, making a NumPy array of them. Then it converts that NumPy array to a Python list, which converts the float32 values to Python `float`. Python's `float` is float64 (this is, in practice, the case on all architectures). Python itself has no built-in float32 type.

That program logic appears in this code from the `openai.Embedding.create` method, from the file `openai/api_resources/get_embedding.py`, which also reveals that not all models are guaranteed support base64 encoding (but I verified the suite of the inner `if` *is* executed when using `text-embedding-ada-002`):

```python
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
```

Since float32 and float64 are *binary* floating point, there is error associated with the base conversion between decimal and binary. For example, in a REPL, `np.float32('0.033652876')` and `float('0.033652876')` both show the string representation `0.033652876`, while `float(np.float32('0.033652876'))` shows the string representation `0.03365287557244301`.

These float64 representations are being produced because Python has no built-in float32 type, not because the extra precision would usually be helpful. But I suppose you might get a *tiny* reduction in computational error by avoiding ever having the values represented in decimal--either ever or until they are converted to float64. You can use the same technique `openai.Embedding.create` uses. When accessing the API endpoint directly, you can request `base64` and decode the results yourself. *How* you should do this depends on what language/framework you are using.

The only thing that makes me a littie uneasy is that I haven't found where `encoding_format` is documented.
