package io.github.eliahkagan.embed_encode;

import java.util.List;

public record Result(String object, List<Embedding> data, String model, Usage usage) {
    public record Embedding(String object, int index, String embedding) {
    }

    public record Usage(int prompt_tokens, int total_tokens) {
    }
}
