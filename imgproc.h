#ifndef _IMGPROC_H
#define _IMGPROC_H
#include <memory.h>
#include <x86intrin.h>

int FastBlur(unsigned char *img, unsigned char *dst, int width, int height, int roi_x, int roi_y, int roi_w, int roi_h, int radius);
int DownSampling2X(unsigned char *src, unsigned char *dst, int width, int height);
int DownSampling4X(unsigned char *src, unsigned char *dst, int width, int height);
int DownSampling8X(unsigned char *src, unsigned char *dst, int width, int height);
void SplitUV(unsigned char *src, unsigned char *dst, int size);
void MergeUV(unsigned char *src, unsigned char *dst, int size);
void Average(unsigned char *src1, unsigned char *src2, unsigned char *dst, int size);

#endif

