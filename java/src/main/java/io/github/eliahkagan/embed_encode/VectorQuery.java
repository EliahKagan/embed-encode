package io.github.eliahkagan.embed_encode;

import java.util.List;

/** Wrapper for a vector (List of floats), supporting several operations. */
final class VectorQuery {
    /** Wrap the given coordinates. */
    public VectorQuery(List<Float> coordinates) {
        _coordinates = coordinates;
    }

    /** Checks if any coordinate of a vector is NaN. */
    public boolean hasNaN() {
        return _coordinates.stream().anyMatch(x -> x.isNaN());
    }

    /** Checks if any coordinate of a vector is infinite. */
    public boolean hasInfinity() {
        return _coordinates.stream().anyMatch(x -> x.isInfinite());
    }

    /** Computes the dot product of the vector with itself. */
    public float normSquared() {
        var computed = _coordinates.stream()
            .mapToDouble(x -> x)
            .map(x -> x * x)
            .sum();

        return (float)computed;
    }

    private final List<Float> _coordinates;
}
