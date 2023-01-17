package io.github.eliahkagan.embed_encode;

import java.util.List;

/**
 * The retrieved embedding, both as original Base64 and as float coordinates.
 * @param base64  The Base64 representation of the embedding.
 * @param coordinates  The "embedding itself": a List of 32-bit floats.
 */
record Base64Embedding(String base64, List<Float> coordinates) {
}
