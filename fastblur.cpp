#include <memory.h>
#include <x86intrin.h>

#define Max(a, b) (((a) > (b)) ? (a) : (b))
#define Min(a, b) (((a) > (b)) ? (b) : (a))

// Convert 4 32-bit integer values to 32 bits (4 unsigned char values).
inline void _mm_storesi128_4uchar(__m128i src, unsigned char *dst)
{
    __m128i T = _mm_packs_epi32(src, src);
    T = _mm_packus_epi16(T, T);
    *((int*)dst) = _mm_cvtsi128_si32(T);
}

// Convert 8 32-bit integer values to 64 bits (8 unsigned char values).
inline void _mm256_storesi256_8uchar(__m256i src, unsigned char *dst)
{
    __m256i T1 = _mm256_packs_epi32(src, _mm256_permute2x128_si256(src, src, 0x01));
    __m128i T2 = _mm_packus_epi16(_mm256_extractf128_si256(T1, 0), _mm_setzero_si128());
    *((int64_t*)dst) = _mm_cvtsi128_si64(T2);
}

// Calculate the sum of 4 32-bit integer values in __m128i
inline int _mm_sumsi128_epi32(__m128i V)                       // 3,    2,  1, 0
{
    __m128i T = _mm_hadd_epi32(V, _mm_setzero_si128());        // 32,   10, 0, 0
    T = _mm_hadd_epi32(T, _mm_setzero_si128());                // 3210, 0,  0, 0
    return _mm_cvtsi128_si32(T);                               // extract low 32 bits
}

inline int GetMirrorPos(int length, int pos)
{
    if (pos < 0)
        return -pos;
    else if (pos >= length)
        return length + length - pos - 2;
    else
        return pos;
}

inline unsigned char ClampToByte(int value)
{
    if (value < 0)
        return 0;
    else if (value > 255)
        return 255;
    else
        return (unsigned char)value;
}

int GetSum(int *data, int length)
{
    register int x = 0;
    int block8_end = length / 8 * 8;
    __m256i sum_v = _mm256_setzero_si256();
    for (; x < block8_end; x += 8)
    {
        sum_v = _mm256_add_epi32(sum_v, _mm256_loadu_si256((__m256i *)(data + x + 0)));
    }
    int sum = _mm_sumsi128_epi32(_mm_add_epi32(_mm256_extracti128_si256(sum_v, 0), _mm256_extracti128_si256(sum_v, 1)));
    for (; x < length; x++)
    {
        sum += data[x];
    }
    return sum;
}

int FastBlur(unsigned char *img, unsigned char *dst, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h, int radius)
{
    if (img == NULL)
        return -1;
    if (width <= 0 || height <= 0 || radius <= 0)
        return -1;
    if (roi_x < 0 || roi_y < 0 || roi_w <= 0 || roi_h <= 0)
        return -1;
    if (roi_x + roi_w > width || roi_y + roi_h > height)
        return -1;

    int boxsize = (2 * radius + 1) * (2 * radius + 1);
    float inv = 1.0f / boxsize;

    int *temp = (int *)_mm_malloc((roi_w + radius + radius) * sizeof(int), 32);
    int *y_offset = (int *)_mm_malloc((height + radius + radius) * sizeof(int), 32);
    if (temp == NULL || y_offset == NULL)
    {
        if (temp != NULL) free(temp);
        if (y_offset != NULL) free(y_offset);
        return -1;
    }

    int inplace = (dst == NULL) ? 1 : 0;
    unsigned char *_dst = NULL;
    if (inplace)
    {
        _dst = (unsigned char *)_mm_malloc((roi_w * roi_h) * sizeof(unsigned char), 32);
    }
    else
    {
        _dst = dst;
    }
    if (_dst == NULL)
        return -1;

    for (int y = 0; y < height + radius + radius; y++)
        y_offset[y] = GetMirrorPos(height, y - radius);

    int x_min = Max(0, radius - roi_x);
    int x_max = Min(width - roi_x + radius, roi_w + radius + radius);
    int block16_end = (x_max - x_min) / 16 * 16 + x_min;
    int block8_end = (x_max - x_min) / 8 * 8 + x_min;
    int block4_end = (x_max - x_min) / 4 * 4 + x_min;

    const __m128 inv_v128 = _mm_set1_ps(inv);
    const __m256 inv_v256 = _mm256_set1_ps(inv);
    const __m128i zero_v128 = _mm_setzero_si128();
    const __m256i zero_v256 = _mm256_setzero_si256();
    const __m256i sll4_v256 = _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7);
    const __m256i sll8_v256 = _mm256_set_epi32(5, 4, 3, 2, 1, 0, 7, 7);
    const __m256i allhigh_v256 = _mm256_set_epi32(7, 7, 7, 7, 7, 7, 7, 7);

    for (int y = 0; y < roi_h; y++)
    {
        unsigned char *pLineDst = inplace ? (_dst + y * roi_w) : (_dst + (roi_y + y) * width + roi_x);
        if (y == 0)  // Get the sum of each pixels in first (radius * 2 + 1) lines
        {
            memset(temp, 0, (roi_w + radius + radius) * sizeof(int));
            for (int z = -radius; z <= radius; z++)
            {
                register int x = x_min;
                unsigned char *pLineSrc = img + y_offset[roi_y + z + radius] * width + roi_x - radius;
                for (; x < block16_end; x += 16)
                {
                    int *pTemp = temp + x;
                    __m256i data_v = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
                    _mm256_storeu_si256((__m256i *)pTemp, _mm256_add_epi32(_mm256_loadu_si256((__m256i *)pTemp), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data_v, 0))));
                    _mm256_storeu_si256((__m256i *)(pTemp + 8), _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(pTemp + 8)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(data_v, 1))));
                }
                for (; x < block8_end; x += 8)
                {
                    int *pTemp = temp + x;
                    __m128i data_v = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)(pLineSrc + x)));
                    _mm_storeu_si128((__m128i *)pTemp, _mm_add_epi32(_mm_loadu_si128((__m128i *)pTemp), _mm_cvtepi16_epi32(data_v)));
                    _mm_storeu_si128((__m128i *)(pTemp + 4), _mm_add_epi32(_mm_loadu_si128((__m128i *)(pTemp + 4)), _mm_unpackhi_epi16(data_v, zero_v128)));
                }
                for (; x < x_max; x++)
                {
                    temp[x] += pLineSrc[x];
                }
            }
        }
        else  // Subtract the earliest line and add a new line to get the sum of each pixels in the next (radius * 2 + 1) lines
        {
            register int x = x_min;
            unsigned char *pLineOut = img + y_offset[roi_y + y - 1] * width + roi_x - radius;              // Line pointer to subtract
            unsigned char *pLineIn = img + y_offset[roi_y + y + radius + radius] * width + roi_x - radius; // Line pointer to add

            for (; x < block16_end; x += 16)
            {
                int *pTemp = temp + x;
                __m256i out_v = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineOut + x)));
                __m256i in_v = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineIn + x)));
                __m256i diff_v = _mm256_sub_epi16(in_v, out_v);
                _mm256_storeu_si256((__m256i *)pTemp, _mm256_add_epi32(_mm256_loadu_si256((__m256i *)pTemp), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff_v, 0))));
                _mm256_storeu_si256((__m256i *)(pTemp + 8), _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(pTemp + 8)), _mm256_cvtepi16_epi32(_mm256_extracti128_si256(diff_v, 1))));
            }
            for (; x < block8_end; x += 8)
            {
                int *pTemp = temp + x;
                __m128i out_v = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)(pLineOut + x)));
                __m128i in_v = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i *)(pLineIn + x)));
                __m128i diff_v = _mm_sub_epi16(in_v, out_v);
                _mm_storeu_si128((__m128i *)pTemp, _mm_add_epi32(_mm_loadu_si128((__m128i *)pTemp), _mm_cvtepi16_epi32(diff_v)));
                _mm_storeu_si128((__m128i *)(pTemp + 4), _mm_add_epi32(_mm_loadu_si128((__m128i *)(pTemp + 4)), _mm_cvtepi16_epi32(_mm_srli_si128(diff_v, 8))));
            }
            for (; x < x_max; x++)
            {
                temp[x] += pLineIn[x] - pLineOut[x];
            }
        }

        if (roi_x - radius < 0)                       // Fill left pixels with mirror data
        {
            register int x = 0;
            register int total_flip = radius - roi_x;
            int block4_end = total_flip / 4 * 4;
            for (; x < block4_end; x += 4)
            {
                __m128i data_v = _mm_loadu_si128((__m128i *)(temp + total_flip + total_flip - x - 3));
                _mm_storeu_si128((__m128i *)(temp + x), _mm_shuffle_epi32(data_v, _MM_SHUFFLE(0, 1, 2, 3)));
            }
            for (; x < total_flip; x++)
            {
                temp[x] = temp[total_flip + total_flip - x];
            }
        }
        if (roi_w + roi_x + radius > width)      // Fill right pixels with mirror data
        {
            register int x = 0;
            register int total_flip = roi_w + roi_x + radius - width;
            int block4_end = total_flip / 4 * 4;
            for (; x < block4_end; x += 4)
            {
                __m128i data_v = _mm_loadu_si128((__m128i *)(temp + total_flip + roi_w - x - 5));
                _mm_storeu_si128((__m128i *)(temp + total_flip + roi_w + x), _mm_shuffle_epi32(data_v, _MM_SHUFFLE(0, 1, 2, 3)));
            }
            for (; x < total_flip; x++)
            {
                temp[total_flip + roi_w + x] = temp[total_flip + roi_w - x - 2];
            }
        }

        int lastsum = GetSum(temp, radius * 2 + 1);  // Get the sum of each pixels in first (radius * 2 + 1) columns
        register int x = 0;
        block8_end = (roi_w - 1) / 8 * 8 + 1;
        block4_end = (roi_w - 1) / 4 * 4 + 1;
        pLineDst[x++] = ClampToByte(lastsum * inv);

        register __m128i oldsum_v128 = _mm_set1_epi32(lastsum);
        register __m256i oldsum_v256 = _mm256_set1_epi32(lastsum);

        // Get the sum of each pixels in the next (radius * 2 + 1) columns
        for (; x < block8_end; x += 8)
        {
            __m256i out_v = _mm256_loadu_si256((__m256i *)(temp + x - 1));
            __m256i in_v = _mm256_loadu_si256((__m256i *)(temp + x + radius + radius));
            __m256i diff_v = _mm256_sub_epi32(in_v, out_v);
            // 7,        6,       5,      4,     3,    2,   1,  0
            __m256i add1_v = _mm256_add_epi32(diff_v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(diff_v, sll4_v256), zero_v256, 0x01));
            // 76,       65,      54,     43,    32,   21,  10, 0
            __m256i add2_v = _mm256_add_epi32(add1_v, _mm256_blend_epi32(_mm256_permutevar8x32_epi32(add1_v, sll8_v256), zero_v256, 0x03));
            // 7654,     6543,    5432,   4321,  3210, 210, 10, 0
            __m256i value_v = _mm256_add_epi32(add2_v, _mm256_permute2x128_si256(add2_v, add2_v, 0x08));
            // 76543210, 6543210, 543210, 43210, 3210, 210, 10, 0
            __m256i newsum_v = _mm256_add_epi32(oldsum_v256, value_v);
            oldsum_v256 = _mm256_permutevar8x32_epi32(newsum_v, allhigh_v256);
           __m256i mean_v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(newsum_v), inv_v256));
            _mm256_storesi256_8uchar(mean_v, pLineDst + x);
        }
        lastsum = _mm256_extract_epi32(oldsum_v256, 0);
        oldsum_v128 = _mm_set1_epi32(lastsum);
        for (; x < block4_end; x += 4)
        {
            __m128i out_v = _mm_loadu_si128((__m128i *)(temp + x - 1));
            __m128i in_v = _mm_loadu_si128((__m128i *)(temp + x + radius + radius));
            __m128i diff_v = _mm_sub_epi32(in_v, out_v);                           // 3,    2,   1,  0
            __m128i add1_v = _mm_add_epi32(diff_v, _mm_slli_si128(diff_v, 4));     // 32,   21,  10, 0
            __m128i value_v = _mm_add_epi32(add1_v, _mm_slli_si128(add1_v, 8));    // 3210, 210, 10, 0
            __m128i newsum_v = _mm_add_epi32(oldsum_v128, value_v);
            oldsum_v128 = _mm_shuffle_epi32(newsum_v, _MM_SHUFFLE(3, 3, 3, 3));
            __m128i mean_v = _mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(newsum_v), inv_v128));
            _mm_storesi128_4uchar(mean_v, pLineDst + x);
        }
        lastsum = _mm_extract_epi32(oldsum_v128, 0);
        for (; x < roi_w; x++)
        {
            lastsum = lastsum - temp[x - 1] + temp[x + radius + radius];
            pLineDst[x] = ClampToByte(lastsum * inv);
        }
    }

    _mm_free(temp);
    _mm_free(y_offset);

    if (inplace)
    {
        for (int y = 0; y < roi_h; y++)
        {
            memcpy(img + (roi_y + y) * width + roi_x, _dst + y * roi_w, roi_w);
        }
        if (_dst != NULL)
            _mm_free(_dst);
    }
    return 0;
}

