/**
 * @file BatchMandelCalculator.h
 * @author David Bayer <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int *calculateMandelbrot();
    void print_data();

private:
    // @TODO add all internal parameters
    int *data;

    void calculateBatch(int, int);
};

#endif