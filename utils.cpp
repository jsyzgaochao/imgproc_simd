#include <memory.h>
#include <x86intrin.h>

void SplitUV(unsigned char *src, unsigned char *dst, int width, int height)
{
    int size = width * height;
    const __m128i shuffle_mask128 = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                                  1, 3, 5, 7, 9, 11, 13, 15);
    for (int y = 0; y < height; y++)
    {
        int x = 0;
        unsigned char *pLineUV = src + y * width * 2;
        unsigned char *pLineU = dst + y * width;
        unsigned char *pLineV = dst + size + y * width;
        for (; x < width / 16 * 16; x += 16)
        {
            __m128i uv1 = _mm_loadu_si128((__m128i *)(pLineUV + x * 2));
            __m128i uv2 = _mm_loadu_si128((__m128i *)(pLineUV + x * 2 + 16));
            __m128i uvs1 = _mm_shuffle_epi8(uv1, shuffle_mask128);
            __m128i uvs2 = _mm_shuffle_epi8(uv2, shuffle_mask128);
            __m128i u = (__m128i)_mm_shuffle_ps((__m128)uvs1, (__m128)uvs2, _MM_SHUFFLE(1, 0, 1, 0));
            __m128i v = (__m128i)_mm_shuffle_ps((__m128)uvs1, (__m128)uvs2, _MM_SHUFFLE(3, 2, 3, 2));
            _mm_storeu_si128((__m128i *)(pLineU + x), u);
            _mm_storeu_si128((__m128i *)(pLineV + x), v);
        }
        for (; x < width; x++)
        {
            pLineU[x] = pLineUV[2*x];
            pLineV[x] = pLineUV[2*x+1];
        }
    }
}

void MergeUV(unsigned char *src, unsigned char *dst, int width, int height)
{
    int size = width * height;
    for (int y = 0; y < height; y++)
    {
        int x = 0;
        unsigned char *pLineUV = dst + y * width * 2;
        unsigned char *pLineU = src + y * width;
        unsigned char *pLineV = src + size + y * width;
        for (; x < width / 32 * 32; x += 32)
        {
            __m256i u = _mm256_loadu_si256((__m256i *)(pLineU + x));
            __m256i v = _mm256_loadu_si256((__m256i *)(pLineV + x));
            __m256i uvs1 = _mm256_unpacklo_epi8(u, v);
            __m256i uvs2 = _mm256_unpackhi_epi8(u, v);
            __m256i uv1 = _mm256_permute2x128_si256(uvs1, uvs2, 0x20);
            __m256i uv2 = _mm256_permute2x128_si256(uvs1, uvs2, 0x31);
            _mm256_storeu_si256((__m256i *)(pLineUV + 2 * x), uv1);
            _mm256_storeu_si256((__m256i *)(pLineUV + 2 * x + 32), uv2);
        }
        for (; x < width / 16 * 16; x += 16)
        {
            __m128i u = _mm_loadu_si128((__m128i *)(pLineU + x));
            __m128i v = _mm_loadu_si128((__m128i *)(pLineV + x));
            __m128i uv1 = _mm_unpacklo_epi8(u, v);
            __m128i uv2 = _mm_unpackhi_epi8(u, v);
            _mm_storeu_si128((__m128i *)(pLineUV + 2 * x), uv1);
            _mm_storeu_si128((__m128i *)(pLineUV + 2 * x + 16), uv2);
        }
        for (; x < width; x++)
        {
            pLineUV[2*x] = pLineU[x];
            pLineUV[2*x+1] = pLineV[x];
        }
    }
}

