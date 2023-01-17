package io.github.eliahkagan.embed_encode;

import java.util.List;

/**
 * Content received from the server in the POST request for the embedding.
 * @param object  The JSON data type of the {@code data} field.
 * @param data  List of embeddings. In our use, there should be exactly one.
 * @param model  The full name of the embedding model.
 * @param usage  Number of tokens processed in the request.
 */
record Result(String object, List<Embedding> data, String model, Usage usage) {
    /**
     * Content specific to a particular embedding returned.
     * @param object  The JSON data type of the {@code embedding} field.
     * @param index  Which input text this embeds. In our use, this is 0.
     * @param embedding  Representation of the actual embedding.
     */
    public record Embedding(String object, int index, String embedding) {
    }

    /**
     * Number of tokens processed in the request.
     * @param prompt_tokens  Number of input tokens.
     * @param total_tokens  Same as {@code prompt_tokens} (no output tokens).
     */
    public record Usage(int prompt_tokens, int total_tokens) {
    }
}
