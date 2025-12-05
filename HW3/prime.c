#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

// -------------------------------
// Configurable Parameters
// -------------------------------
#define NUM_THREADS 4 
#define MAX_TRIES 1000  // Increased to ensure we find enough primes
#define MAXVAL 99999999999999999UL
#define NUM_VALUES 10   // Total primes to collect

// -------------------------------
// Global Shared Data with Synchronization
// -------------------------------
unsigned long int prime_values[NUM_VALUES];
int values_count = 0;  // Current number of primes collected
pthread_mutex_t values_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t values_condition = PTHREAD_COND_INITIALIZER;
int cancel_requested = 0;  // Flag to signal threads to stop

// -------------------------------
// Barrier Structure and Functions
// -------------------------------
struct barrier {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int num_threads;
    int count;
};

void barrier_init(struct barrier* b, int nt) {
    pthread_mutex_init(&b->lock, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count = 0;
    b->num_threads = nt;
}

void barrier_destroy(struct barrier* b) {
    pthread_mutex_destroy(&b->lock);
    pthread_cond_destroy(&b->cond);
}

void barrier_wait(struct barrier* b) {
    pthread_mutex_lock(&b->lock);
    b->count++;

    if (b->count == b->num_threads) {
        pthread_cond_broadcast(&b->cond);
    } else {
        while (b->count < b->num_threads) {
            pthread_cond_wait(&b->cond, &b->lock);
        }
    }
    pthread_mutex_unlock(&b->lock);
}

// -------------------------------
// Thread Argument Structure
// -------------------------------
struct thread_args {
    int rank;
    struct barrier* barrier;
};

// -------------------------------
// Primality Test (Brute Force)
// -------------------------------
int is_prime(unsigned long int n) {
    if (n < 2) return 0;
    if (n % 2 == 0 && n != 2) return 0;

    unsigned long int limit = (unsigned long int)sqrt((long double)n);
    for (unsigned long int i = 3; i <= limit; i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

// -------------------------------
// Add Prime to Global Array (Thread-Safe)
// -------------------------------
int add_prime(unsigned long int prime) {
    pthread_mutex_lock(&values_mutex);
    
    // Check if we've already collected enough primes
    if (values_count >= NUM_VALUES) {
        pthread_mutex_unlock(&values_mutex);
        return 0;  // Array full
    }
    
    // Add prime to array
    prime_values[values_count] = prime;
    values_count++;
    
    // Check if we've reached the target
    int reached_target = (values_count == NUM_VALUES);
    
    pthread_mutex_unlock(&values_mutex);
    
    // Signal master thread if target reached
    if (reached_target) {
        pthread_cond_signal(&values_condition);
    }
    
    return 1;  // Successfully added
}

// -------------------------------
// Thread Function
// -------------------------------
void* start(void* x) {
    struct thread_args* args = (struct thread_args*) x;
    unsigned long int to_test = 0;
    
    printf("Thread %d started searching for primes\n", args->rank);

    for (int i = 0; i < MAX_TRIES && !cancel_requested; i++) {
        // Generate a random unsigned long int
        to_test = ((unsigned long int)rand() << 32) | rand();
        to_test = to_test % MAXVAL;

        if (is_prime(to_test)) {
            printf("Thread %d found prime: %lu\n", args->rank, to_test);
            
            // Try to add prime to global array
            if (add_prime(to_test)) {
                printf("Thread %d successfully added prime to array\n", args->rank);
            } else {
                printf("Thread %d could not add prime (array full)\n", args->rank);
                break;  // Array is full, stop searching
            }
        }
        
        // Small delay to prevent excessive CPU usage
        struct timespec ts = {0, 1000000};  // 1ms
        nanosleep(&ts, NULL);
    }

    printf("Thread %d exiting\n", args->rank);
    pthread_exit(NULL);
}

// -------------------------------
// Main Function
// -------------------------------
int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    struct thread_args args[NUM_THREADS];
    struct barrier sb;
    struct timespec start_time, end_time;
    
    // Start timing
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // Seed the random number generator
    srand(time(NULL));

    // Initialize global array
    for (int i = 0; i < NUM_VALUES; i++) {
        prime_values[i] = 0;
    }

    barrier_init(&sb, NUM_THREADS);

    printf("Starting %d threads to find %d primes...\n", NUM_THREADS, NUM_VALUES);

    // Create all threads
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].rank = i;
        args[i].barrier = &sb;
        pthread_create(&threads[i], NULL, start, &args[i]);
    }

    // Master thread waits for condition variable
    pthread_mutex_lock(&values_mutex);
    while (values_count < NUM_VALUES) {
        printf("Master: Waiting for primes (%d/%d collected)...\n", values_count, NUM_VALUES);
        pthread_cond_wait(&values_condition, &values_mutex);
    }
    pthread_mutex_unlock(&values_mutex);
    
    printf("Master: Target reached! Cancelling threads...\n");
    
    // Signal all threads to stop
    cancel_requested = 1;
    
    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    // End timing
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    // Calculate execution time
    double execution_time = (end_time.tv_sec - start_time.tv_sec) + 
                           (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;

    // Print results
    printf("\n=== RESULTS ===\n");
    printf("Threads used: %d\n", NUM_THREADS);
    printf("Primes collected: %d\n", values_count);
    printf("Target primes: %d\n", NUM_VALUES);
    printf("Total execution time: %.6f seconds\n", execution_time);
    
    printf("\nCollected primes:\n");
    for (int i = 0; i < values_count; i++) {
        printf("Prime[%d] = %lu\n", i, prime_values[i]);
    }

    // Cleanup
    barrier_destroy(&sb);
    pthread_mutex_destroy(&values_mutex);
    pthread_cond_destroy(&values_condition);
    
    return 0;
}
