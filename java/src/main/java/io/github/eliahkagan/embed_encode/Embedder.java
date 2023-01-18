// Copyright (c) 2023 Eliah Kagan
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
// SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
// OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
// CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

package io.github.eliahkagan.embed_encode;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

/**
 * High-level interface to text-embedding-ada-002 in the OpenAI embeddings API.
 * <p>
 *   Embeddings are fetched as Base64 and provided as {@link Base64Embedding},
 *   which provides both the original Base64 string and the vector coordinates
 *   it represents.
 * </p>
 */
final class Embedder {
    /**
     * Creates a new Embedder using the given API key as its bearer token.
     * @param apiKey  The OpenAI API key. (Should start with "sk-".)
     */
    public Embedder(String apiKey) {
        _apiKey = apiKey;
    }

    /**
     * Produces a text embedding.
     * @param text  The text to embed.
     * @return The embedding, as Base64 and as 32-bit float coordinates.
     * @throws IOException  If the request fails or can't be attempted.
     */
    public Base64Embedding embed(String text) throws IOException {
        // Query the OpenAI API for binary data (transmitted as Base64).
        var result = queryApi(text);
        var base64 = result.data().get(0).embedding();
        var bytes = Base64.getDecoder().decode(base64);

        // View the raw binary data as floats (IEEE 754 binary32).
        var byteBuffer = ByteBuffer.wrap(bytes);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        var floatBuffer = byteBuffer.asFloatBuffer();
        assertDimension(floatBuffer.remaining());

        // Copy the floats in the buffer to a List representing an embedding.
        List<Float> coordinates = new ArrayList<>(floatBuffer.remaining());
        while (floatBuffer.hasRemaining()) {
            coordinates.add(floatBuffer.get());
        }

        return new Base64Embedding(base64, coordinates);
    }

    /** Expected dimension for text-embedding-ada-002. */
    private static final int EXPECTED_DIMENSION = 1536;

    /** Content-Type for the HTTP request. */
    private static final MediaType JSON
        = MediaType.get("application/json; charset=utf-8");

    /**
     * Stops execution if the dimension is wrong for text-embedding-ada-002.
     * @param actualDimension  The dimension to validate.
     */
    private static void assertDimension(int actualDimension) {
        if (actualDimension == EXPECTED_DIMENSION) return;

        var message = String.format(
            "expected dimension %d, got %d",
            EXPECTED_DIMENSION,
            actualDimension);

        throw new AssertionError(message);
    }

    /**
     * Queries the API. This high-level function takes care of object mapping.
     * @param text  The text to embed.
     * @return The result returned by the API, represented as a {@code Result}.
     * @throws IOException  If the request fails or can't be attempted.
     */
    private Result queryApi(String text) throws IOException {
        var mapper = new ObjectMapper();
        var data = new RequestData(text, "text-embedding-ada-002", "base64");
        var body = RequestBody.create(mapper.writeValueAsString(data), JSON);
        return mapper.readValue(queryApiRaw(body), Result.class);
    }

    /**
     * Queries the API. Lower-level helper for @{code queryApi}.
     * @param body  The encoded JSON request body with content-type metadata.
     * @return The raw JSON content from the response, as a string.
     * @throws IOException  If the request fails or can't be attempted.
     */
    private String queryApiRaw(RequestBody body) throws IOException {
        var client = new OkHttpClient();

        var request = new Request.Builder()
            .url("https://api.openai.com/v1/embeddings")
            .header("Authorization", "Bearer " + _apiKey)
            .post(body)
            .build();

        try (var response = client.newCall(request).execute()) {
            // FIXME: Can body() be null or would IOException have been thrown?
            return response.body().string();
        }
    }

    /** The user's OpenAI API key, to use as a bearer token in the request. */
    private final String _apiKey;
}
