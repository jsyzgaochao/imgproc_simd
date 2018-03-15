#include <memory.h>
#include <x86intrin.h>

int DownSampling2X(unsigned char *src, unsigned char *dst, int width, int height)
{
    if (src == NULL || dst == NULL)
        return -1;
    short *temp = (short *)_mm_malloc(width * sizeof(short), 32);
    for (int y = 0; y < height - 1; y++)
    {
        int x = 0;
        int block16 = width / 16;
        int block32 = width / 32;
        // init with 1st line
        unsigned char *pLineSrc = src + y * width;
        for (; x < block32 * 32; x += 32)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_loadu_si256((__m256i *)(pLineSrc + x));
            _mm256_storeu_si256((__m256i *)pLineDst, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 0)));
            _mm256_storeu_si256((__m256i *)(pLineDst + 16), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1)));
        }
        for (; x < block16 * 16; x += 16)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
            _mm256_storeu_si256((__m256i *)pLineDst, data);
        }
        for (; x < width; x++)
        {
            temp[x] = pLineSrc[x];
        }
        // calc sum of 2 lines
        pLineSrc = src + (y + 1) * width;
        x = 0;
        for (; x < block32 * 32; x += 32)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_loadu_si256((__m256i *)(pLineSrc + x));
            _mm256_storeu_si256((__m256i *)pLineDst, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)pLineDst), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 0))));
            _mm256_storeu_si256((__m256i *)(pLineDst + 16), _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(pLineDst + 16)), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1))));
        }
        for (; x < block16 * 16; x += 16)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
            _mm256_storeu_si256((__m256i *)pLineDst, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)pLineDst), data));
        }
        for (; x < width; x++)
        {
            temp[x] = pLineSrc[x];
        }
        // calc sum of 2 rows and get avg value
        x = 0;
        for (; x < block32 * 32; x += 32)
        {
            unsigned char *pLineDst = dst +  (width / 2) * (y / 2) + x / 2;
            __m256i A = _mm256_loadu_si256((__m256i *)(temp + x));
            __m256i B = _mm256_loadu_si256((__m256i *)(temp + x + 16));
            __m256i AB = _mm256_permute4x64_epi64(_mm256_hadd_epi16(A, B), 0xD8);
            AB = _mm256_srai_epi16(AB, 2);
            __m128i ABx = _mm256_extracti128_si256(_mm256_packus_epi16(AB, _mm256_permute2x128_si256(AB,AB,0x01)), 0);
            _mm_storeu_si128((__m128i *)pLineDst, ABx);
        }
        for (; x < block16 * 16; x += 16)
        {
            unsigned char *pLineDst = dst +  (width / 2) * (y / 2) + x / 2;
            __m128i A = _mm_loadu_si128((__m128i *)(temp + x));
            __m128i B = _mm_loadu_si128((__m128i *)(temp + x + 8));
            __m128i AB = _mm_hadd_epi16(A, B);
            AB = _mm_srai_epi16(AB, 2);
            *((int64_t*)pLineDst) = _mm_cvtsi128_si64(_mm_packus_epi16(AB, _mm_setzero_si128()));
        }
        for (; x < width - 1; x += 2)
        {
            unsigned char *pLineDst = dst +  (width / 2) * (y / 2) + x / 2;
            int sum = temp[x+0] + temp [x+1];
            *pLineDst = sum >> 2;
        }
    }
    _mm_free(temp);
    return 0;
}

int DownSampling4X(unsigned char *src, unsigned char *dst, int width, int height)
{
    if (src == NULL || dst == NULL)
        return -1;
    short *temp = (short *)_mm_malloc(width * sizeof(short), 16);
    for (int y = 0; y < height - 3; y+=4)
    {
        int x = 0;
        int block16 = width / 16;
        int block32 = width / 32;
        int block64 = width / 64;
        // init with 1st line
        unsigned char *pLineSrc = src + y * width;
        for (; x < block32 * 32; x += 32)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_loadu_si256((__m256i *)(pLineSrc + x));
            _mm256_storeu_si256((__m256i *)pLineDst, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 0)));
            _mm256_storeu_si256((__m256i *)(pLineDst + 16), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1)));
        }
        for (; x < block16 * 16; x += 16)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
            _mm256_storeu_si256((__m256i *)pLineDst, data);
        }
        for (; x < width; x++)
        {
            temp[x] = pLineSrc[x];
        }
        // calc sum of 4 lines
        for (int z = 1; z < 4; z++)
        {
            unsigned char *pLineSrc = src + (y + z) * width;
            int x = 0;
            for (; x < block32 * 32; x += 32)
            {
                short *pLineDst = temp + x;
                __m256i data = _mm256_loadu_si256((__m256i *)(pLineSrc + x));
                _mm256_storeu_si256((__m256i *)pLineDst, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)pLineDst), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 0))));
                _mm256_storeu_si256((__m256i *)(pLineDst + 16), _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(pLineDst + 16)), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1))));
            }
            for (; x < block16 * 16; x += 16)
            {
                short *pLineDst = temp + x;
                __m256i data = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
                _mm256_storeu_si256((__m256i *)pLineDst, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)pLineDst), data));
            }
            for (; x < width; x++)
            {
                temp[x] = pLineSrc[x];
            }
        }
        // calc sum of 4 rows and get avg value
        x = 0;
        for (; x < block64 * 64; x += 64)
        {
            unsigned char *pLineDst = dst +  (width / 4) * (y / 4) + x / 4;
            __m256i A = _mm256_loadu_si256((__m256i *)(temp + x));
            __m256i B = _mm256_loadu_si256((__m256i *)(temp + x + 16));
            __m256i C = _mm256_loadu_si256((__m256i *)(temp + x + 32));
            __m256i D = _mm256_loadu_si256((__m256i *)(temp + x + 48));
            __m256i AB = _mm256_permute4x64_epi64(_mm256_hadd_epi16(A, B), 0xD8);
            __m256i CD = _mm256_permute4x64_epi64(_mm256_hadd_epi16(C, D), 0xD8);
            __m256i ABCD = _mm256_permute4x64_epi64(_mm256_hadd_epi16(AB, CD), 0xD8);
            ABCD = _mm256_srai_epi16(ABCD, 4);
            __m128i ABCDx = _mm256_extracti128_si256(_mm256_packus_epi16(ABCD, _mm256_permute2x128_si256(ABCD,ABCD,0x01)), 0);
            _mm_storeu_si128((__m128i *)pLineDst, ABCDx);
        }
        for (; x < block16 * 16; x += 16)
        {
            unsigned char *pLineDst = dst +  (width / 4) * (y / 4) + x / 4;
            __m128i A = _mm_loadu_si128((__m128i *)(temp + x));
            __m128i B = _mm_loadu_si128((__m128i *)(temp + x + 8));
            __m128i AB = _mm_hadd_epi16(_mm_hadd_epi16(A, B), _mm_setzero_si128());
            AB = _mm_srai_epi16(AB, 4);
            *((int32_t*)pLineDst) = _mm_cvtsi128_si32(_mm_packus_epi16(AB, _mm_setzero_si128()));
        }
        for (; x < width - 3; x += 4)
        {
            unsigned char *pLineDst = dst +  (width / 4) * (y / 4) + x / 4;
            int sum = temp[x+0] + temp[x+1] + temp[x+2] + temp[x+3];
            *pLineDst = sum >> 4;
        }
    }
    _mm_free(temp);
    return 0;
}

int DownSampling8X(unsigned char *src, unsigned char *dst, int width, int height)
{
    if (src == NULL || dst == NULL)
        return -1;
    short *temp = (short *)_mm_malloc(width * sizeof(short), 16);
    for (int y = 0; y < height - 7; y+=8)
    {
        int x = 0;
        int block16 = width / 16;
        int block32 = width / 32;
        int block128 = width / 128;
        // init with 1st line
        unsigned char *pLineSrc = src + y * width;
        for (; x < block32 * 32; x += 32)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_loadu_si256((__m256i *)(pLineSrc + x));
            _mm256_storeu_si256((__m256i *)pLineDst, _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 0)));
            _mm256_storeu_si256((__m256i *)(pLineDst + 16), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1)));
        }
        for (; x < block16 * 16; x += 16)
        {
            short *pLineDst = temp + x;
            __m256i data = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
            _mm256_storeu_si256((__m256i *)pLineDst, data);
        }
        for (; x < width; x++)
        {
            temp[x] = pLineSrc[x];
        }
        // calc sum of 8 lines
        for (int z = 1; z < 8; z++)
        {
            unsigned char *pLineSrc = src + (y + z) * width;
            int x = 0;
            for (; x < block32 * 32; x += 32)
            {
                short *pLineDst = temp + x;
                __m256i data = _mm256_loadu_si256((__m256i *)(pLineSrc + x));
                _mm256_storeu_si256((__m256i *)pLineDst, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)pLineDst), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 0))));
                _mm256_storeu_si256((__m256i *)(pLineDst + 16), _mm256_add_epi16(_mm256_loadu_si256((__m256i *)(pLineDst + 16)), _mm256_cvtepu8_epi16(_mm256_extracti128_si256(data, 1))));
            }
            for (; x < block16 * 16; x += 16)
            {
                short *pLineDst = temp + x;
                __m256i data = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pLineSrc + x)));
                _mm256_storeu_si256((__m256i *)pLineDst, _mm256_add_epi16(_mm256_loadu_si256((__m256i *)pLineDst), data));
            }
            for (; x < width; x++)
            {
                temp[x] = pLineSrc[x];
            }
        }
        // calc sum of 8 rows and get avg value
        x = 0;
        for (; x < block128 * 128; x += 128)
        {
            unsigned char *pLineDst = dst +  (width / 8) * (y / 8) + x / 8;
            __m256i A = _mm256_loadu_si256((__m256i *)(temp + x));
            __m256i B = _mm256_loadu_si256((__m256i *)(temp + x + 16));
            __m256i C = _mm256_loadu_si256((__m256i *)(temp + x + 32));
            __m256i D = _mm256_loadu_si256((__m256i *)(temp + x + 48));
            __m256i E = _mm256_loadu_si256((__m256i *)(temp + x + 64));
            __m256i F = _mm256_loadu_si256((__m256i *)(temp + x + 80));
            __m256i G = _mm256_loadu_si256((__m256i *)(temp + x + 96));
            __m256i H = _mm256_loadu_si256((__m256i *)(temp + x + 112));
            __m256i AB = _mm256_permute4x64_epi64(_mm256_hadd_epi16(A, B), 0xD8);
            __m256i CD = _mm256_permute4x64_epi64(_mm256_hadd_epi16(C, D), 0xD8);
            __m256i EF = _mm256_permute4x64_epi64(_mm256_hadd_epi16(E, F), 0xD8);
            __m256i GH = _mm256_permute4x64_epi64(_mm256_hadd_epi16(G, H), 0xD8);
            __m256i ABCD = _mm256_permute4x64_epi64(_mm256_hadd_epi16(AB, CD), 0xD8);
            __m256i EFGH = _mm256_permute4x64_epi64(_mm256_hadd_epi16(EF, GH), 0xD8);
            __m256i ABCDEFGH = _mm256_permute4x64_epi64(_mm256_hadd_epi16(ABCD, EFGH), 0xD8);
            ABCDEFGH = _mm256_srai_epi16(ABCDEFGH, 6);
            __m128i ABCDEFGHX = _mm256_extracti128_si256(_mm256_packus_epi16(ABCDEFGH, _mm256_permute2x128_si256(ABCDEFGH,ABCDEFGH,0x01)), 0);
            _mm_storeu_si128((__m128i *)pLineDst, ABCDEFGHX);
        }
        for (; x < block32 * 32; x += 32)
        {
            unsigned char *pLineDst = dst +  (width / 8) * (y / 8) + x / 8;
            __m256i A = _mm256_loadu_si256((__m256i *)(temp + x));
            __m256i B = _mm256_loadu_si256((__m256i *)(temp + x + 16));
            __m256i AB = _mm256_permute4x64_epi64(_mm256_hadd_epi16(A, B), 0xD8);
            AB = _mm256_permute4x64_epi64(_mm256_hadd_epi16(AB, _mm256_setzero_si256()), 0xD8);
            AB = _mm256_permute4x64_epi64(_mm256_hadd_epi16(AB, _mm256_setzero_si256()), 0xD8);
            AB = _mm256_srai_epi16(AB, 6);
            __m128i ABx = _mm256_extracti128_si256(_mm256_packus_epi16(AB, _mm256_permute2x128_si256(AB,AB,0x01)), 0);
            *((int32_t*)pLineDst) = _mm_cvtsi128_si32(ABx);
        }
        for (; x < width - 7; x += 8)
        {
            unsigned char *pLineDst = dst +  (width / 8) * (y / 8) + x / 8;
            int sum = temp[x+0] + temp[x+1] + temp[x+2] + temp[x+3] + temp[x+4] + temp[x+5] + temp[x+6] + temp[x+7];
            *pLineDst = sum >> 6;
        }
    }
    _mm_free(temp);
    return 0;
}

