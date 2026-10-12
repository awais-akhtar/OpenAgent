/* Original paper Eq. (8)-(10) source aggregation, not a K3 decoder.
 * Row-major sources [count,width], query [width], optional key gain [width].
 * Caller supplies sized disjoint buffers. Errors leave output unchanged.
 */
#include <math.h>
#include <stddef.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int oa_kimi_k3_attnres(const float *sources, const float *query,
                            const float *gain, int count, int width,
                            double epsilon, float *output)
{
    double scores[128], staged[4096], maximum, denominator = 0.0;
    int i, j;
    if (!sources || !query || !output || count < 1 || count > 128 ||
        width < 1 || width > 4096 || !isfinite(epsilon) || epsilon <= 0.0)
        return 1;
    for (j = 0; j < width; ++j) {
        if (!isfinite(query[j]) || (gain && !isfinite(gain[j]))) return 2;
        staged[j] = 0.0;
    }
    for (i = 0; i < count; ++i) {
        double square = 0.0, score = 0.0, inverse;
        for (j = 0; j < width; ++j) {
            double x = sources[(size_t)i * (size_t)width + (size_t)j];
            if (!isfinite(x)) return 2;
            square += x * x;
        }
        inverse = 1.0 / sqrt(square / width + epsilon);
        for (j = 0; j < width; ++j)
            score += (double)query[j] * sources[(size_t)i * (size_t)width + (size_t)j]
                     * inverse * (gain ? gain[j] : 1.0);
        if (!isfinite(score)) return 2;
        scores[i] = score;
    }
    maximum = scores[0];
    for (i = 1; i < count; ++i) if (scores[i] > maximum) maximum = scores[i];
    /* Normalize keys only; the original unnormalized sources remain values. */
    for (i = 0; i < count; ++i) {
        double weight = exp(scores[i] - maximum);
        denominator += weight;
        for (j = 0; j < width; ++j)
            staged[j] += weight * sources[(size_t)i * (size_t)width + (size_t)j];
    }
    for (j = 0; j < width; ++j) {
        staged[j] /= denominator;
        if (!isfinite(staged[j])) return 2;
    }
    for (j = 0; j < width; ++j) output[j] = (float)staged[j];
    return 0;
}
