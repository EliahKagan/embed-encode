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

/** Wrapper for a vector (List of floats), supporting several operations. */
final class VectorOperations {
    /** Wraps the given coordinates. */
    public VectorOperations(List<Float> coordinates) {
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

    /** Gets the coordinates as double-precision values. */
    public List<Double> doubles() {
        return _coordinates.stream()
            .map(Double::valueOf)
            .toList();
    }

    private final List<Float> _coordinates;
}
