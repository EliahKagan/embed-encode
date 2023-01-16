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
import java.util.Base64;
import java.util.List;
import java.util.stream.IntStream;

public class Main {
    public static void main(String[] args) throws IOException {
        var text = getTextToEmbed(args);
        System.out.println(text);

        var embedding = embed(text);
        System.out.println(embedding);
        System.out.format("has NaN = %s%n", hasNaN(embedding));
        System.out.format("norm squared = %f%n", normSquared(embedding));
    }

    private static int DIMENSION = 1536;

    private static final MediaType JSON
        = MediaType.get("application/json; charset=utf-8");

    private static String getTextToEmbed(String[] args) {
        return args.length == 0
            ? "The food was delicious and the waiter..." // As in OpenAI docs.
            : String.join(" ", args);
    }

    private static List<Float> embed(String text) throws IOException {
        var result = queryApi(text);
        var base64 = result.data().get(0).embedding();
        var bytes = Base64.getDecoder().decode(base64);

        var buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        return IntStream.range(0, DIMENSION)
            .mapToObj(buffer::getFloat)
            .toList();
    }

    private static Result queryApi(String text) throws IOException {
        var mapper = new ObjectMapper();
        var data = new RequestData(text, "text-embedding-ada-002", "base64");
        var body = RequestBody.create(mapper.writeValueAsString(data), JSON);
        return mapper.readValue(queryApiRaw(body), Result.class);
    }

    private static String queryApiRaw(RequestBody body) throws IOException {
        var client = new OkHttpClient();

        var request = new Request.Builder()
            .url("https://api.openai.com/v1/embeddings")
            .header("Authorization", "Bearer " + getApiKey())
            .post(body)
            .build();

        try (var response = client.newCall(request).execute()) {
            return response.body().string();
        }
    }

    private static String getApiKey() throws IOException {
        var key = System.getenv("OPENAI_API_KEY");

        if (key == null || key.isBlank()) {
            var path = Path.of("..", ".api_key");
            key = Files.readString(path, StandardCharsets.UTF_8);
        }

        return key.strip();
    }

    private static boolean hasNaN(List<Float> vector) {
        return vector.stream().anyMatch(x -> x.isNaN());
    }

    private static float normSquared(List<Float> vector) {
        var computed = vector.stream()
            .mapToDouble(x -> x)
            .map(x -> x * x)
            .sum();

        return (float)computed;
    }
}