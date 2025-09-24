#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Function to fill an array with random values scaled by "scale"
void random_array(double* array, int size, double scale) {
    for (int i = 0; i < size; i++) {
        array[i] = ((double)rand() / RAND_MAX) * scale;
    }
}

// Function to compute the sum of an array
double sum(double* array, int size) {
    double s = 0.0;
    for (int i = 0; i < size; i++) {
        s += array[i];
    }
    return s;
}

// Function to compute the standard deviation of an array
double stdev(double* array, int size) {
    if (size <= 1) return 0.0;

    double mean = sum(array, size) / size;
    double variance = 0.0;

    for (int i = 0; i < size; i++) {
        double diff = array[i] - mean;
        variance += diff * diff;
    }
    variance /= (size - 1);  // sample standard deviation

    return sqrt(variance);
}

// Function to smooth an array using weighted average of neighbors
void smooth(double* array, int size, double w) {
    if (size <= 2) return; // no smoothing possible with fewer than 3 elements

    double* temp = (double*)malloc(size * sizeof(double));
    if (!temp) {
        fprintf(stderr, "Memory allocation failed in smooth()\n");
        exit(EXIT_FAILURE);
    }

    // Keep the first and last unchanged
    temp[0] = array[0];
    temp[size - 1] = array[size - 1];

    for (int i = 1; i < size - 1; i++) {
        double neighbor_avg = (array[i - 1] + array[i + 1]) / 2.0;
        temp[i] = array[i] * w + neighbor_avg * (1.0 - w);
    }

    // Copy back
    for (int i = 0; i < size; i++) {
        array[i] = temp[i];
    }

    free(temp);
}

// ---- Main program to test everything ----
int main(void) {
    srand((unsigned int)time(NULL));

    int size = 20;
    double scale = 10.0;
    double w = 0.6; // smoothing weight
    int iterations = 5;

    double* arr = (double*)malloc(size * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Memory allocation failed in main()\n");
        return EXIT_FAILURE;
    }

    // Fill with random values
    random_array(arr, size, scale);

    printf("Initial array:\n");
    for (int i = 0; i < size; i++) {
        printf("%6.3f ", arr[i]);
    }
    printf("\n");

    double sd = stdev(arr, size);
    printf("Initial standard deviation: %.5f\n", sd);

    // Apply smoothing iteratively
    for (int it = 1; it <= iterations; it++) {
        smooth(arr, size, w);
        sd = stdev(arr, size);
        printf("After smoothing %d: stdev = %.5f\n", it, sd);
    }

    free(arr);
    return 0;
}

