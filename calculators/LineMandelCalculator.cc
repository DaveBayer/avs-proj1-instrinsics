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
/*
#include <thread>
#include "ThreadPool.h"
*/
#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	// @TODO allocate & prefill memory

	data = (int *) _mm_malloc(height * width * sizeof(int), 4);
	if (data = nullptr) {
		std::cerr << "Bad alloc" << std::endl;
		exit(1);
	}
}

LineMandelCalculator::~LineMandelCalculator()
{
	// @TODO cleanup the memory

	_mm_free(data);
	data = nullptr;
}

void LineMandelCalculator::print_data()
{
	for (int i = 0; i < height; i++) {
		std::cout << i << ":\t";

		for (int j = 0; j < width; j++) {
			std::cout << data[i * width + j] << " ";
		}

		std::cout << std::endl;
	}
}

static inline __attribute__((always_inline))
__m512i mandelbrot(__m512 real, __m512 imag, int limit, __mmask16 mask)
{
	__m512i result = _mm512_setzero_epi32();
	__mmask16 result_mask = mask;

	const __m512 two = _mm512_set1_ps(2.f);
	const __m512 four = _mm512_set1_ps(4.f);

	__m512 zReal = real;
	__m512 zImag = imag;

	for (int i = 0; i < limit; i++) {
	//	r2 = zReal * zReal
		const __m512 r2 = _mm512_mul_ps(zReal, zReal);

	//	i2 = zImag * zImag
		const __m512 i2 = _mm512_mul_ps(zImag, zImag);

	//	if (r2 + i2 > 4.0f) then write i to result and update result_mask
		__mmask16 test_mask = _mm512_cmp_ps_mask(_mm512_add_ps(r2, i2), four, _CMP_GT_OS);

		result = _mm512_mask_mov_epi32(result, test_mask & result_mask, _mm512_set1_epi32(i));
		result_mask &= ~test_mask;

		if (result_mask == 0x0000U)
			return result;
		
	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm512_fmadd_ps(two, _mm512_mul_ps(zReal, zImag), imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm512_add_ps(_mm512_sub_ps(r2, i2), real);
	}

	result = _mm512_mask_mov_epi32(result, result_mask, _mm512_set1_epi32(limit));

	return result;
}

static inline __attribute__((always_inline))
__m512 _mm512_concat_ps256(__m256 a, __m256 b)
{
	__m512 x;

	x = _mm512_castps256_ps512(a);
	x = _mm512_mask_broadcast_f32x8(x, 0xff00, b);

	return x;
}

#define AVX512_SIZE_PD 8
#define AVX512_SIZE_PS 16

int *LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int *pdata = data;

	__m512d dx_pd, x_start_pd, inc_pd;

	dx_pd = _mm512_set1_pd(dx);
	x_start_pd = _mm512_set1_pd(x_start);

	inc_pd = _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.);

	for (int i = 0; i < height; i++) {
	//	y = y_start + i * dy
		const __m512 y = _mm512_set1_ps(y_start + i * dy);

		for (int j = 0; j < width; j += AVX512_SIZE_PS) {
		//	prepare j
			__m512d j1_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j)), inc_pd);
			__m512d j2_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j + AVX512_SIZE_PD)), inc_pd);

		//	x = x_start + j * dx
			__m512d x1_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j1_pd, dx_pd));
			__m512d x2_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j2_pd, dx_pd));
			__m512 x = _mm512_concat_ps256(_mm512_cvtpd_ps(x1_pd), _mm512_cvtpd_ps(x2_pd));

		//	get mask for avx instructions and data pointer increment
			__mmask16 mask = 0xffffU;
			int inc = AVX512_SIZE_PS;

			int diff = width - j;
			
			if (diff < AVX512_SIZE_PS) {
				mask >>= AVX512_SIZE_PS - diff;
				inc = diff;
			}

			__m512i values = mandelbrot(x, y, limit, mask);

			std::cout << "Storing.. " << pdata << std::endl;
		//	store values in memory pointed by pdata using mask
			_mm512_mask_storeu_epi32(pdata, mask, values);

			pdata += inc;
		}
	}


	return data;
}

/*
void mandelbrot_line(int line_num, int *pdata, int width, double dx, double x_start, double dy, double y_start, int limit)
{
	__m512d dx_pd, x_start_pd, inc_pd;

	dx_pd = _mm512_set1_pd(dx);
	x_start_pd = _mm512_set1_pd(x_start);

	inc_pd = _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.);

//	y = y_start + i * dy
	const __m512 y = _mm512_set1_ps(y_start + line_num * dy);

	for (int j = 0; j < width; j += AVX512_SIZE_PS) {
	//	prepare j
		__m512d j1_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j)), inc_pd);
		__m512d j2_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j + AVX512_SIZE_PD)), inc_pd);

	//	x = x_start + j * dx
		__m512d x1_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j1_pd, dx_pd));
		__m512d x2_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j2_pd, dx_pd));
		__m512 x = _mm512_concat_ps256(_mm512_cvtpd_ps(x1_pd), _mm512_cvtpd_ps(x2_pd));

	//	get mask for avx instructions and data pointer increment
		__mmask16 mask = 0xffffU;
		int inc = AVX512_SIZE_PS;

		int diff = width - j;
		
		if (diff < AVX512_SIZE_PS) {
			mask >>= AVX512_SIZE_PS - diff;
			inc = diff;
		}

		__m512i values = mandelbrot(x, y, limit, mask);


	//	store values in memory pointed by pdata using mask
		_mm512_mask_store_epi32(pdata, mask, values);

		pdata += inc;
	}
}

int *LineMandelCalculator::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers

	int max_threads = std::thread::hardware_concurrency();

	ThreadPool thread_pool(max_threads);

	int *pdata = data;

	for (int i = 0; i < height; i++) {
		thread_pool.enqueue(mandelbrot_line, i, pdata, width, dx, x_start, dy, y_start, limit);
		pdata += width;
	}

	return data;
}
*/