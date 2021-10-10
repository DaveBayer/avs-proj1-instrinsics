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

static inline __m512i mandelbrot(__m512 real, __m512 imag, int limit, __mmask16 mask)
{
	__m512i result = _mm512_setzero_epi32();
	__mmask16 result_mask = mask;

	const __m512 two = _mm512_set1_ps(2.f);
	const __m512 four = _mm512_set1_ps(4.f);

	__m512 zReal = real;
	__m512 zImag = imag;

	float tmp[16] = { 0.f };

	for (int i = 0; i < limit; i++) {
		const __m512 r2 = _mm512_mul_ps(zReal, zReal);
		const __m512 i2 = _mm512_mul_ps(zImag, zImag);

		_mm512_storeu_ps(tmp, _mm512_add_ps(r2, i2));

	//	if (r2 + i2 > 4.0f) then write i to result
		__mmask16 test_mask = _mm512_cmp_ps_mask(_mm512_add_ps(r2, i2), four, _CMP_GT_OS);

		for (int i = 0; i < 16; i++) {
			if ((1U << i) & mask)
				std::cout << tmp[i] << " " << ((test_mask >> i) & 1U) << "\t";
		}
		
		std::cout << std::endl;

		result = _mm512_mask_mov_epi32(result, test_mask & result_mask, _mm512_set1_epi32(i));
		//	__mmask16 res_mask_old = result_mask;
		result_mask &= ~test_mask;

		//	_mm512_mask_storeu_epi32(tmp, ~result_mask, result);

		//	std::cout << std::dec << i << std::hex << ": rm.old: " << res_mask_old << " tm: "  << test_mask << " rm: " << result_mask << "\t";


		if (result_mask == 0x0000U)
			return result;

	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm512_fmadd_ps(two, _mm512_mul_ps(zReal, zImag), imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm512_sub_ps(r2, _mm512_add_ps(i2, real));
	}

	result = _mm512_mask_mov_epi32(result, result_mask, _mm512_set1_epi32(limit));

	return result;
	
}

static inline __m512 _mm512_concat_ps256(__m256 a, __m256 b)
{
	__m512 x;

	x = _mm512_castps256_ps512(a);
	x = _mm512_mask_broadcast_f32x8(x, 0xff00, b);

	return x;
}

static inline __m512i _mm512_concat_i256(__m256i a, __m256i b)
{
	__m512i x;

	x = _mm512_castsi256_si512(a);
	x = _mm512_mask_broadcast_i32x8(x, 0xff00, b);

	return x;
}

int * LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers

	const int AVX512_SIZE_PS = 16;
	const int AVX512_SIZE_PD = 8;

	__m512d dx_pd, x_start_pd, inc_pd;

	dx_pd = _mm512_set1_pd(dx);
	x_start_pd = _mm512_set1_pd(x_start);

	inc_pd = _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.);

	for (int i = 0, *row_ptr = data; i < height; i++, row_ptr += width) {

	//	y = y_start + i * dy
		__m512 y = _mm512_set1_ps(y_start + i * dy);

		for (int j = 0, *col_ptr = row_ptr; j < width; j += AVX512_SIZE_PS, col_ptr += AVX512_SIZE_PS) {
			int diff = width - j;
			__mmask16 mask = 0xffffU;
			
			if (diff < AVX512_SIZE_PS)
				mask >>= AVX512_SIZE_PS - diff;

			__m512d j1_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j)), inc_pd);
			__m512d j2_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j + AVX512_SIZE_PD)), inc_pd);

		//	x = x_start + j * dx
			__m512d x1_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j1_pd, dx_pd));
			__m512d x2_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j2_pd, dx_pd));
			__m512 x = _mm512_concat_ps256(_mm512_cvtpd_ps(x1_pd), _mm512_cvtpd_ps(x2_pd));

			__m512i values = mandelbrot(x, y, limit, mask);


		//	store values in memory pointed by col_ptr using mask
			_mm512_mask_storeu_epi32(col_ptr, mask, values);
		}

		std::cout << std::endl;
	}
/*
	for (int i = 0; i < height; i++) {
		std::cout << std::dec << i << ":\t";

		for (int j = 0; j < width; j++)
			std::cout << std::dec << data[i * width + j] << " ";

		std::cout << std::endl;
	}
*/
	return data;
}
