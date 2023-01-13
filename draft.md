It looks like the embeddings are float32 (single precision). Decimal representations of float32 values have about 7 significant figures. The ~10-digit representations from the API have 8 digits if leading zeros are omitted (i.e., they are decimal representations that have 8-digit mantissas if rewritten in scientific notation). So although the API endpoint isn't returning a whole lot of extra information, the further extra digits you get when using the OpenAI Python library are not significant.

The extra digits arise from a 2-step conversion where representations from the API endpoint are are parsed as float32, and then those float32 values are subsequently converted to float64 (double precision). If the representations from the API endpoint are directly parsed as float64, the extra digits are absent. Python's `float` type is float64 (in practice, this is the case on all architectures). Python doesn't have a built-in float32 type, but NumPy has it. When the OpenAI Python library is used to get embeddings, and no encoding format is specified, part of what it does is to use NumPy for parsing, calling `np.frombuffer` and passing `dtype="float32"`.

As a tiny example of this effect by evaluating expressions in a REPL, `np.float32('0.033652876')` and `float('0.033652876')` both show the string representation `0.033652876`, while `float(np.float32('0.033652876'))` shows the string representation `0.03365287557244301`. Note that I am *not* saying that the two-step conversion has more computational error. Assuming parsing it as float32 is a round-trip conversion from the binary representation the model produced, that step will never increase error. It makes sense that the OpenAI Python library parses them in that way.

I should say what I looked at in the code that leads me to these conclusions. If I understand you correctly, you're using `openai.embeddings_utils.get_embedding`. That function, as shown in `openai/embeddings_utils.py`, is a higher level interface that calls `openai.Embedding.create`, and it does not pass an `encoding_format` argument. As shown in `openai/api_resources/embedding.py`, when `openai.Embedding.create` is called without an `encoding_format` argument, it requests base64-encoded data, then parses and converts it this way:
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
I don't *think* I can link to GitHub in this channel, but this code can be viewed on openai/openai-python (and maybe through a feature of your editor/IDE). To confirm my understanding, I checked that the assignment statement with the `np.frombuffer` call actually runs when calling `openai.Embedding.create` without passing an `encoding_format`. (It does.)
