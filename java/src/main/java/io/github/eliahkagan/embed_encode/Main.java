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

import com.fasterxml.jackson.core.util.DefaultIndenter;
import com.fasterxml.jackson.core.util.DefaultPrettyPrinter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * This program embeds some text, showing Base64 and coordinates.
 *
 * <p>The following output is displayed:</p>
 * <ul>
 *   <li>The text to be embedded.</li>
 *   <li>The retrieved Base64 string that encodes the embedding.</li>
 *   <li>The embedding, as a {@code List<Float>}.</li>
 *   <li>Some stats to help check if the result is plausibly correct.</li>
 *   <li>The embedding, converted to a {@code List<Double>}.</li>
 * </ul>
 *
 * <p>
 *   Furthermore, the {@code List<Double>} is dumped as JSON, to
 *   {@code java-embedding.json}, for comparison to data produced by the OpenAI
 *   Python library.
 * </p>
 */
public class Main {
    /**
     * Program entry point.
     * @param args  Joined to produce custom text to embed, if supplied.
     * @throws IOException  If the embedding request fails or can't be tried
     *                      (such as if the OpenAI API key can't be found).
     */
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
     * Prints out the Base64 string and the List of floats, on separate lines.
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
     * Reports details useful to checking if the embedding makes sense.
     * <p>
     *   Embeddings are real-valued vectors, and thus have no NaN or infinite
     *   coordinates. Furthermore, OpenAI embeddings are normalized (though the
     *   norm, and thus its square, may differ slightly from 1, due to
     *   rounding error).
     * </p>
     * @param coordinates  The float coordinates of the embedding.
     */
    private static void reportPlausibilityDetails(List<Float> coordinates) {
        var vecOps = new VectorOperations(coordinates);
        System.out.format("has NaN?  %b%n", vecOps.hasNaN());
        System.out.format("has infinity?  %b%n", vecOps.hasInfinity());
        System.out.format("norm squared = %.8f%n", vecOps.normSquared());
    }

    /**
     * Converts coordinates to a List of doubles, show them, and save as JSON.
     * <p>This is to compare to Python results. See python/ada-002.ipynb.</p>
     * @param coordinates  The float coordinates of the embedding.
     * @throws IOException  If the json file cannot be written.
     */
    private static void
    reportDoubles(List<Float> coordinates) throws IOException {
        var doubles = new VectorOperations(coordinates).doubles();
        System.out.println(doubles);

        var printer = new DefaultPrettyPrinter()
            .withArrayIndenter(new DefaultIndenter("    ", "\n"));

        new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT)
            .writer(printer)
            .writeValue(new File("java-embedding.json"), doubles);
    }
}
