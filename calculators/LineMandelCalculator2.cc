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
#include "LineMandelCalculator2.h"


LineMandelCalculator2::LineMandelCalculator2 (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator2")
{
	// @TODO allocate & prefill memory

	data = (int *) _mm_malloc(height * width * sizeof(int), 4);
	if (data == nullptr) {
		std::cerr << "Bad alloc" << std::endl;
		exit(1);
	}
}

LineMandelCalculator2::~LineMandelCalculator2()
{
	// @TODO cleanup the memory

	_mm_free(data);
	data = nullptr;
}

void LineMandelCalculator2::print_data()
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
__m256i mandelbrot(__m256 real, __m256 imag, int limit, __mmask8 mask)
{
	__m256i result = _mm256_setzero_si256();
	__mmask8 result_mask = mask;

	const __m256 two = _mm256_set1_ps(2.f);
	const __m256 four = _mm256_set1_ps(4.f);

	__m256 zReal = real;
	__m256 zImag = imag;

	for (int i = 0; i < limit; i++) {
	//	r2 = zReal * zReal
		const __m256 r2 = _mm256_mul_ps(zReal, zReal);

	//	i2 = zImag * zImag
		const __m256 i2 = _mm256_mul_ps(zImag, zImag);

	//	if (r2 + i2 > 4.0f) then write i to result and update result_mask
		__mmask8 test_mask = _mm256_cmp_ps_mask(_mm256_add_ps(r2, i2), four, _CMP_GT_OS);

		result = _mm256_mask_mov_epi32(result, test_mask & result_mask, _mm256_set1_epi32(i));
		result_mask &= ~test_mask;

		if (result_mask == 0x00U)
			return result;
		
	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm256_fmadd_ps(two, _mm256_mul_ps(zReal, zImag), imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm256_add_ps(_mm256_sub_ps(r2, i2), real);
	}

	result = _mm256_mask_mov_epi32(result, result_mask, _mm256_set1_epi32(limit));

	return result;
}

#define AVX256_SIZE_PD 4
#define AVX256_SIZE_PS 8

int *LineMandelCalculator2::calculateMandelbrot () {
	// @TODO implement the calculator & return array of integers
	int *pdata = data;

	__m512d dx_pd, x_start_pd, inc_pd;

	dx_pd = _mm512_set1_pd(dx);
	x_start_pd = _mm512_set1_pd(x_start);

	inc_pd = _mm512_set_pd(7., 6., 5., 4., 3., 2., 1., 0.);

	for (int i = 0; i < height; i++) {
	//	y = y_start + i * dy
		const __m256 y = _mm256_set1_ps(y_start + i * dy);

		for (int j = 0; j < width; j += AVX256_SIZE_PS) {
			__m512d j_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j)), inc_pd);

		//	x = x_start + j * dx
			__m512d x_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j_pd, dx_pd));
			__m256 x = _mm512_cvtpd_ps(x_pd);

		//	get mask for avx instructions and data pointer increment
			__mmask8 mask = 0xffU;
			int inc = AVX256_SIZE_PS;

			int diff = width - j;
			
			if (diff < AVX256_SIZE_PS) {
				mask >>= AVX256_SIZE_PS - diff;
				inc = diff;
			}

			__m256i values = mandelbrot(x, y, limit, mask);


		//	store values in memory pointed by pdata using mask
			_mm256_mask_storeu_epi32(pdata, mask, values);

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
		_mm512_mask_storeu_epi32(pdata, mask, values);

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