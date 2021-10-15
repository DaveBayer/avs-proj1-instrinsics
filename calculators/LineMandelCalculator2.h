/**
 * @file LineMandelCalculator.h
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator2 : public BaseMandelCalculator
{
public:
    LineMandelCalculator2(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator2();
    int *calculateMandelbrot();
//    int *calculateMandelbrot2();
    void print_data();

private:
    // @TODO add all internal parameters
    int *data;
};