#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// ---------------------------------------------
// Random array fill with thread-safe rand_r()
// ---------------------------------------------
void random_array(double* array, int size, double scale) {
    double t0 = omp_get_wtime();

    #pragma omp parallel default(none) shared(array, size, scale)
    {
        unsigned int seed = 1234 + omp_get_thread_num(); // thread-local seed

        #pragma omp for
        for (int i = 0; i < size; i++) {
            array[i] = ((double)rand_r(&seed) / RAND_MAX) * scale;
        }
    }

    double t1 = omp_get_wtime();
    printf("Time (random_array): %.6f sec\n", t1 - t0);
}

// ---------------------------------------------
// Compute sum (with reduction)
// ---------------------------------------------
double sum(double* array, int size) {
    double s = 0.0;
    #pragma omp parallel for default(none) shared(array, size) reduction(+:s)
    for (int i = 0; i < size; i++) {
        s += array[i];
    }
    return s;
}

// ---------------------------------------------
// Compute standard deviation (parallelized)
// ---------------------------------------------
double stdev(double* array, int size) {
    double t0 = omp_get_wtime();

    if (size <= 1) return 0.0;

    double mean = sum(array, size) / size;
    double variance = 0.0;

    #pragma omp parallel for default(none) shared(array, size, mean) reduction(+:variance)
    for (int i = 0; i < size; i++) {
        double diff = array[i] - mean;
        variance += diff * diff;
    }
    variance /= (size - 1);
    double st = sqrt(variance);

    double t1 = omp_get_wtime();
    printf("Time (stdev): %.6f sec\n", t1 - t0);

    return st;
}

// ---------------------------------------------
// Smooth using neighbors (safe, out-of-place)
// ---------------------------------------------
void smooth(double* array, int size, double w) {
    double t0 = omp_get_wtime();

    if (size <= 2) return;

    double* temp = (double*)malloc(size * sizeof(double));
    if (!temp) {
        fprintf(stderr, "Memory allocation failed in smooth()\n");
        exit(EXIT_FAILURE);
    }

    temp[0] = array[0];
    temp[size - 1] = array[size - 1];

    #pragma omp parallel for default(none) shared(array, temp, size, w)
    for (int i = 1; i < size - 1; i++) {
        double neighbor_avg = (array[i - 1] + array[i + 1]) / 2.0;
        temp[i] = array[i] * w + neighbor_avg * (1.0 - w);
    }

    #pragma omp parallel for default(none) shared(array, temp, size)
    for (int i = 0; i < size; i++) {
        array[i] = temp[i];
    }

    free(temp);

    double t1 = omp_get_wtime();
    printf("Time (smooth): %.6f sec\n", t1 - t0);
}

// ---------------------------------------------
// Main program
// ---------------------------------------------
int main(void) {
    srand((unsigned int)time(NULL));

    // Adjust array size for scaling experiments
    int size = 100000000;  // 1e8 elements
    double scale = 10.0;
    double w = 0.6;
    int iterations = 5;

    double* arr = (double*)malloc(size * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed in main()\n");
        return EXIT_FAILURE;
    }

    printf("Array size = %d\n", size);
    printf("Threads = %d\n", omp_get_max_threads());
    printf("--------------------------------------------\n");

    double total_start = omp_get_wtime();

    random_array(arr, size, scale);

    double sd = stdev(arr, size);
    printf("Initial stdev = %.5f\n", sd);
    printf("--------------------------------------------\n");

    for (int it = 1; it <= iterations; it++) {
        smooth(arr, size, w);
        sd = stdev(arr, size);
        printf("After smoothing %d: stdev = %.5f\n", it, sd);
        printf("--------------------------------------------\n");
    }

    double total_end = omp_get_wtime();
    printf("Total time (all steps): %.3f sec\n", total_end - total_start);

    free(arr);
    return 0;
}

