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
