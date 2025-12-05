#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <time.h>

// -------------------------------
// Configurable Parameters
// -------------------------------
#define NUM_THREADS 100 
#define MAX_TRIES 10
#define MAXVAL 99999999999999999UL

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
// Thread Function
// -------------------------------
void* start(void* x) {
    struct thread_args* args = (struct thread_args*) x;
    unsigned long int to_test = 0;
    int found = 0;

    for (int i = 0; i < MAX_TRIES; i++) {
        // Generate a random unsigned long int
        to_test = ((unsigned long int)rand() << 32) | rand();
        to_test = to_test % MAXVAL;

        if (is_prime(to_test)) {
            found = 1;
            break;
        }
    }

    // Barrier synchronization before reporting results
    barrier_wait(args->barrier);

    if (found) {
        printf("thread %d reports that %lu is prime\n", args->rank, to_test);
        pthread_exit((void*)to_test);
    } else {
        printf("thread %d reports no prime\n", args->rank);
        pthread_exit((void*)0);
    }
}

// -------------------------------
// Main Function
// -------------------------------
int main(int argc, char *argv[]) {
    pthread_t threads[NUM_THREADS];
    struct thread_args args[NUM_THREADS];
    struct barrier sb;

    // Seed the random number generator
    srand(time(NULL));

    barrier_init(&sb, NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].rank = i;
        args[i].barrier = &sb;
        pthread_create(&threads[i], NULL, start, &args[i]);
    }

    // Collect thread return values
    for (int i = 0; i < NUM_THREADS; i++) {
        void* retval;
        pthread_join(threads[i], &retval);
        // Optionally store retval for later
    }

    barrier_destroy(&sb);
    return 0;
}

