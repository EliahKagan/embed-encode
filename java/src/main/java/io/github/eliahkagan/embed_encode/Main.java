package io.github.eliahkagan.embed_encode;

import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        var text = getTextToEmbed(args);
        System.out.println(text);

        var embedding = embed(text);
        System.out.println(embedding);
        System.out.format("has NaN?  %b%n", hasNaN(embedding));
        System.out.format("has infinity?  %b%n", hasInfinity(embedding));
        System.out.format("norm squared = %.8f%n", normSquared(embedding));
    }

    /** Expected dimension for text-embedding-ada-002. */
    private static final int EXPECTED_DIMENSION = 1536;

    /** Content-Type for the HTTP request. */
    private static final MediaType JSON
        = MediaType.get("application/json; charset=utf-8");

    /**
     * Joins command-line arguments to text, or uses the fallback text.
     * <p>
     *   If no arguments were passed, then the short text shown in the OpenAI
     *   <a href="https://beta.openai.com/docs/api-reference/embeddings/create">Create embeddings</a>
     *   example is used.
     * </p>
     * @param args  The command-line arguments that were passed to the program.
     */
    private static String getTextToEmbed(String[] args) {
        return args.length == 0
            ? "The food was delicious and the waiter..."
            : String.join(" ", args);
    }

    /**
     * Produces a text embedding, represented as a List of 32-bit floats.
     * @param text  The text to embed.
     * @return The embedding, as a List of coordinates represented as floats.
     * @throws IOException  If the request fails or can't be attempted.
     */
    private static List<Float> embed(String text) throws IOException {
        // Query the OpenAI API for binary data (transmitted as Base64).
        var result = queryApi(text);
        var base64 = result.data().get(0).embedding();
        System.out.println(base64); // FIXME: Remove after debugging.
        var bytes = Base64.getDecoder().decode(base64);

        // View the raw binary data as floats (IEEE 754 binary32).
        var byteBuffer = ByteBuffer.wrap(bytes);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        var floatBuffer = byteBuffer.asFloatBuffer();
        assertDimension(floatBuffer.remaining());

        // Copy the floats in the buffer to a List representing an embedding.
        List<Float> embedding = new ArrayList<>(floatBuffer.remaining());
        while (floatBuffer.hasRemaining()) {
            embedding.add(floatBuffer.get());
        }
        return embedding;
    }

    /**
     * Queries the API. This high-level function takes care of object mapping.
     * @param text  The text to embed.
     * @return The result returned by the API, represented as a {@code Result}.
     * @throws IOException  If the request fails or can't be attempted.
     */
    private static Result queryApi(String text) throws IOException {
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
    private static String queryApiRaw(RequestBody body) throws IOException {
        var client = new OkHttpClient();

        var request = new Request.Builder()
            .url("https://api.openai.com/v1/embeddings")
            .header("Authorization", "Bearer " + getApiKey())
            .post(body)
            .build();

        try (var response = client.newCall(request).execute()) {
            // FIXME: Can body() be null or would IOException have been thrown?
            return response.body().string();
        }
    }

    /**
     * Gets the OpenAI API key. Looks in $OPENAI_API_KEY and ../.api_key.
     * @return The API key, to be used as a bearer token for the request.
     * @throws IOException  If the key file is needed but can't be read.
     */
    private static String getApiKey() throws IOException {
        var key = System.getenv("OPENAI_API_KEY");

        if (key == null || key.isBlank()) {
            var path = Path.of("..", ".api_key");
            key = Files.readString(path, StandardCharsets.UTF_8);
        }

        return key.strip();
    }

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

    /** Checks if any coordinate of a vector is NaN. */
    private static boolean hasNaN(List<Float> vector) {
        return vector.stream().anyMatch(x -> x.isNaN());
    }

    /** Checks if any coordinate of a vector is infinite. */
    private static boolean hasInfinity(List<Float> vector) {
        return vector.stream().anyMatch(x -> x.isInfinite());
    }

    /** Computes the dot product of a vector with itself. */
    private static float normSquared(List<Float> vector) {
        var computed = vector.stream()
            .mapToDouble(x -> x)
            .map(x -> x * x)
            .sum();

        return (float)computed;
    }
}