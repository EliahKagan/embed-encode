package io.github.eliahkagan.embed_encode;

/**
 * Content sent to the server in the HTTP POST request for an embedding.
 * @param input  The text whose embedding should be computed.
 * @param model  The embedding model to use.
 * @param encoding_format  How the embedding should be encoded.
 */
record RequestData(String input, String model, String encoding_format) {
}
