package io.github.eliahkagan.embed_encode;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class Main {
    public static void main(String[] args) throws IOException {
        var text = getTextToEmbed(args);
        System.out.println(text);

        var embedding = new Embedder(getApiKey()).embed(text);
        showEmbedding(embedding);
        reportPlausibilityDetails(embedding.coordinates());
        reportDoubles(embedding.coordinates());
    }

    /**
     * Joins command-line arguments to produce text, or uses the fallback text.
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
     * Print out the Base64 string and the List of floats, on separate lines.
     * <p>
     *   {@code Base64Embedding.toString} formats them differently, which I
     *   don't want to change, because it is good for debugging.
     * </p>
     * @param embedding  The Base64 and {@code List<Float>} data to print.
     */
    private static void showEmbedding(Base64Embedding embedding) {
        System.out.println(embedding.base64());
        System.out.println(embedding.coordinates());
    }

    /**
     * Report details useful to checking if the embedding makes sense.
     * <p>
     *   Embeddings are real-valued vectors, and thus have no NaN or infinite
     *   coordinates. Furthermore, OpenAI embeddings are normalized (though the
     *   norm, and thus its square, may differ slightly from 1, due to
     *   rounding error).
     * </p>
     * @param coordinates  The float coordinates of the embedding.
     */
    private static void reportPlausibilityDetails(List<Float> coordinates) {
        var vecQuery = new VectorQuery(coordinates);
        System.out.format("has NaN?  %b%n", vecQuery.hasNaN());
        System.out.format("has infinity?  %b%n", vecQuery.hasInfinity());
        System.out.format("norm squared = %.8f%n", vecQuery.normSquared());
    }

    /**
     * Convert coordinates to a List of doubles, show them, and save as JSON.
     * <p>This is to compare to Python results. See python/ada-002.ipynb.</p>
     * @param coordinates  The float coordinates of the embedding.
     */
    private static void reportDoubles(List<Float> coordinates) {
        var doubles = new VectorQuery(coordinates).doubles();
        System.out.println(doubles);
    }
}
