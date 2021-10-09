/**
 * @file LineMandelCalculator.cc
 * @author David Bayer <xbayer09@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 2021-10-9
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <immintrin.h>


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory

	data = new int[height * width];
}

LineMandelCalculator::~LineMandelCalculator() {
	// @TODO cleanup the memory

	delete[] data;
	data = nullptr;
}

static inline __m512i mandelbrot(__m512 real, __m512 imag, int limit)
{
	__m512i result = _mm512_setzero_epi32();
	__mmask16 result_mask = 0U;
	const __mmask16 target_mask = 0xffffU;

	const __m512 two = _mm512_set1_ps(2.f);
	const __m512 four = _mm512_set1_ps(4.f);

	__m512 zReal = real;
	__m512 zImag = imag;

	for (int i = 0; i < limit; i++) {
		__m512 r2 = _mm512_mul_ps(zReal, zReal);
		__m512 i2 = _mm512_mul_ps(zImag, zImag);

	//	if (r2 + i2 > 4.0f) then write i to result
		__mmask16 test_mask = _mm512_cmp_ps_mask(_mm512_mul_ps(r2, i2), four, _CMP_GT_OQ);

		result = _mm512_mask_mov_epi32(result, test_mask ^ result_mask, _mm512_set1_epi32(i));
		result_mask |= test_mask;

		if (result_mask == target_mask)
			break;

	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm512_mul_ps(two, _mm512_mul_ps(zReal, zImag));
		zImag = _mm512_add_ps(zImag, imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm512_sub_ps(r2, i2);
		zReal = _mm512_add_ps(zReal, real);
	}

	result = _mm512_mask_mov_epi32(result, result_mask ^ target_mask, _mm512_set1_epi32(limit));

	return result;
}

#define _MM512_FILL_INCREMENTS_PD(from) _mm512_set_pd(from, from + 1.0, from + 2.0, from + 3.0, from + 4.0, from + 5.0, from + 6.0, from + 7.0)

static inline __m512 _mm512_concat_ps256(__m256 a, __m256 b)
{
	__m512 x;

	x = _mm512_castps256_ps512(b);
	x = _mm512_mask_broadcast_f32x8(x, 0xff00, a);

	return x;
}

int * LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers

	int *pdata = data;
	const int AVX512_SIZE_PS = 16;
	const int AVX512_SIZE_PD = 8;

	__m512d dx_pd, dy_pd, x_start_pd, y_start_pd;

	dx_pd = _mm512_set1_pd(dx);
	dy_pd = _mm512_set1_pd(dy);

	x_start_pd = _mm512_set1_pd(x_start);
	y_start_pd = _mm512_set1_pd(y_start);

	for (int i = 0; i < height; i++) {
		const __m512d i_pd = _mm512_set1_pd(i);

		for (int j = 0; j < width; j += AVX512_SIZE_PS) {
			const __m512d j1_pd = _MM512_FILL_INCREMENTS_PD(j);
			const __m512d j2_pd = _MM512_FILL_INCREMENTS_PD(j + AVX512_SIZE_PD);

			__m512d x1_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j1_pd, dx_pd));
			__m512d x2_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j2_pd, dx_pd));
			__m512 x = _mm512_concat_ps256(_mm512_cvtpd_ps(x1_pd), _mm512_cvtpd_ps(x2_pd));

			__m512d y_pd = _mm512_add_pd(y_start_pd, _mm512_mul_pd(i_pd, dx_pd));
			__m256 y_ps = _mm512_cvtpd_ps(y_pd);			
			__m512 y = _mm512_concat_ps256(y_ps, y_ps);

			__m512i values = mandelbrot(x, y, limit);
			_mm512_mask_storeu_epi32(pdata, (1 << (j % AVX512_SIZE_PS)) - 1, values);

			pdata += AVX512_SIZE_PS;
		}
	}

	return data;
}
