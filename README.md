Image processing with SIMD optimization
==========
Some image processing method with SSE/AVX optimization.

Functions:
----------
```
int FastBlur(unsigned char *img, unsigned char *dst, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h, int radius);
int DownSampling2X(unsigned char *src, unsigned char *dst, int width, int height);
int DownSampling4X(unsigned char *src, unsigned char *dst, int width, int height);
int DownSampling8X(unsigned char *src, unsigned char *dst, int width, int height);
void SplitUV(unsigned char *src, unsigned char *dst, int size);
void MergeUV(unsigned char *src, unsigned char *dst, int size);
void Average(unsigned char *src1, unsigned char *src2, unsigned char *dst, int size);
```

Test:
----------
Test environment: Intel Core i3-6100 @ 3.70GHz

The input image has the resolution of 1920*1090 and only 1 channel. The function is executed in single thread.

|Function      |Time       |
|--------------|-----------|
|FastBlur      |995 us     |
|DownSampling2X|241 us     |
|DownSampling4X|102 us     |
|DownSampling8X|92 us      |
|SplitUV       |50 us      |
|MergeUV       |40 us      |
|Average       |235 us     |

Part of the algorithm and code refer to Imageshop (http://www.cnblogs.com/Imageshop/) , and use AVX/AVX2 instruction set to optimize the code.
