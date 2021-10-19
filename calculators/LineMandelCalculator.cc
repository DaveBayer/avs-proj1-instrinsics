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
#include <exception>


#include <stdlib.h>
#include <immintrin.h>
/*
#include <thread>
#include "ThreadPool.h"
*/
#include "LineMandelCalculator.h"

#undef __AVX512F__

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#	pragma message("Using AVX512F & AVX512DQ")
#	define MM_ALIGNMENT 64
#	define MM_SIZE_32BIT 16
#	define MM_SIZE_64BIT 8
#elif defined(__AVX__) && defined(__AVX2__)
#	pragma message("Using AVX & AVX2")
#	define MM_ALIGNMENT 32
#	define MM_SIZE_32BIT 8
#	define MM_SIZE_64BIT 4
#else
#	error Unsupported architecture, minimum requirements: AVX, AVX2
#endif

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int *) _mm_malloc(height * width * sizeof(int), MM_ALIGNMENT);
	
	if (data == nullptr)
		throw std::bad_alloc();
}

LineMandelCalculator::~LineMandelCalculator()
{
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

#if defined(__AVX512F__) && defined(__AVX512DQ__)

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
		zImag = _mm512_fmadd_ps(_mm512_mul_ps(two, zReal), zImag, imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm512_add_ps(_mm512_sub_ps(r2, i2), real);
	}

	return _mm512_mask_mov_epi32(result, result_mask, _mm512_set1_epi32(limit));
}

static inline __attribute__((always_inline))
__m512 mm512_concat_ps256(__m256 a, __m256 b)
{
	__m512 x;

	x = _mm512_castps256_ps512(a);
	x = _mm512_mask_broadcast_f32x8(x, 0xff00, b);

	return x;
}

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

		for (int j = 0; j < width; j += MM_SIZE_32BIT) {
		//	prepare j
			__m512d j1_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j)), inc_pd);
			__m512d j2_pd = _mm512_add_pd(_mm512_set1_pd(static_cast<double>(j + MM_SIZE_64BIT)), inc_pd);

		//	x = x_start + j * dx
			__m512d x1_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j1_pd, dx_pd));
			__m512d x2_pd = _mm512_add_pd(x_start_pd, _mm512_mul_pd(j2_pd, dx_pd));
			__m512 x = mm512_concat_ps256(_mm512_cvtpd_ps(x1_pd), _mm512_cvtpd_ps(x2_pd));

		//	get mask for avx instructions and data pointer increment
			__mmask16 mask = 0xffffU;
			int inc = MM_SIZE_32BIT;

			int diff = width - j;
			
			if (diff < MM_SIZE_32BIT) {
				mask >>= MM_SIZE_32BIT - diff;
				inc = diff;
			}

			__m512i values = mandelbrot(x, y, limit, mask);

		//	store values in memory pointed by pdata using mask
			_mm512_mask_storeu_epi32(pdata, mask, values);

			pdata += inc;
		}
	}

	return data;
}

#else

static inline __attribute__((always_inline))
int mandelbrot(float real, float imag, int limit)
{
	float zReal = real;
	float zImag = imag;

	for (int i = 0; i < limit; ++i)
	{
		float r2 = zReal * zReal;
		float i2 = zImag * zImag;

		if (r2 + i2 > 4.0f)
			return i;

		zImag = 2.0f * zReal * zImag + imag;
		zReal = r2 - i2 + real;
	}

	return limit;
}

static inline __attribute__((always_inline))
__m256i mandelbrot_mm256(__m256 real, __m256 imag, int limit)
{
	__m256i result = _mm256_setzero_si256();
	__m256i result_mask = _mm256_set1_epi32(-1);

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
		__m256i test_mask = _mm256_castps_si256(_mm256_cmp_ps(_mm256_add_ps(r2, i2), four, _CMP_GT_OS));

		result = _mm256_blendv_epi8(result, _mm256_set1_epi32(i), _mm256_and_si256(test_mask, result_mask));
		result_mask = _mm256_andnot_si256(test_mask, result_mask);

		if (_mm256_testz_si256(result_mask, _mm256_setzero_si256()));
			return result;
		
	//	zImag = 2.0f * zReal * zImag + imag;
		zImag = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(two, zReal), zImag), imag);

	//	zReal = r2 - i2 + real;
		zReal = _mm256_add_ps(_mm256_sub_ps(r2, i2), real);
	}

	return _mm256_blendv_epi8(result, _mm256_set1_epi32(limit), result_mask);;
}

int *LineMandelCalculator::calculateMandelbrot()
{
	int *pdata = data;

	__m256d dx_pd, x_start_pd, inc_pd;

	dx_pd = _mm256_set1_pd(dx);
	x_start_pd = _mm256_set1_pd(x_start);

	inc_pd = _mm256_set_pd(3., 2., 1., 0.);

	for (int i = 0; i < height; i++) {
	//	y = y_start + i * dy
		const __m256 y = _mm256_set1_ps(y_start + i * dy);

		int j;

		for (j = 0; j + MM_SIZE_32BIT < width; j += MM_SIZE_32BIT) {
		//	prepare j
			__m256d j1_pd = _mm256_add_pd(_mm256_set1_pd(static_cast<double>(j)), inc_pd);
			__m256d j2_pd = _mm256_add_pd(_mm256_set1_pd(static_cast<double>(j + MM_SIZE_64BIT)), inc_pd);

		//	x = x_start + j * dx
			__m256d x1_pd = _mm256_add_pd(x_start_pd, _mm256_mul_pd(j1_pd, dx_pd));
			__m256d x2_pd = _mm256_add_pd(x_start_pd, _mm256_mul_pd(j2_pd, dx_pd));
			__m256 x = _mm256_setr_m128(_mm256_cvtpd_ps(x1_pd), _mm256_cvtpd_ps(x2_pd));

			__m256i values = mandelbrot_mm256(x, y, limit);

		//	store values in memory pointed by pdata using mask
			_mm256_storeu_si256((__m256i_u *) pdata, values);

			pdata += MM_SIZE_32BIT;
		}

		for (; j < width; j++) {
			float x = x_start + j * dx;
			float y = y_start + i * dy;

			int value = mandelbrot(x, y, limit);

			*(pdata++) = value;
		}
	}


	return data;
}

#endif

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