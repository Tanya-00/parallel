#define STEPS 100000000
#define MIN 1
#define MAX 300
#define SEED 100
#define CACHE_LINE 64u

#include <thread>
#include <vector>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include "iostream"

using namespace std;

typedef double (*f_t)(double);
typedef double (*fib_t)(double);

typedef double (*I_t)(f_t, double, double);
typedef unsigned (*F_t)(unsigned);
typedef double (*R_t)(unsigned*, size_t);

typedef struct experiment_result {
    double result;
    double time_ms;
} experiment_result;

#if defined(__GNUC__) && __GNUC__ <= 10
namespace std {
    constexpr size_t hardware_constructive_interference_size = 64u;
    constexpr size_t hardware_destructive_interference_size = 64u;
}
#endif

size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

double f(double x) {
    return x * x;
}

unsigned g_num_threads = thread::hardware_concurrency();

void set_num_threads(unsigned T) {
    omp_set_num_threads(T);
    g_num_threads = T;
}

unsigned get_num_threads() {
    return g_num_threads;
}

experiment_result run_experiment(I_t I) {
    double t0 = omp_get_wtime();
    double R = I(f, -1, 1);
    double t1 = omp_get_wtime();
    return {R, t1 - t0};
}

experiment_result run_experiment_random(R_t R) {
    size_t len = 100000;
    unsigned arr[len];
    unsigned seed = 100;

    double t0 = omp_get_wtime();
    double Res = R((unsigned *)&arr, len);
    double t1 = omp_get_wtime();
    return {Res, t1 - t0};
}

void show_experiment_result(I_t I) {
    double T1;
    printf("%10s\t%10s\t%10sms\t%13s\n", "Threads", "Result", "Time", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment(I);
        if (T == 1) {
            T1 = R.time_ms;
        }
        printf("%10u\t%10g\t%10g\t%10g\n", T, R.result, R.time_ms, T1/R.time_ms);
    };
}

void show_experiment_result_Rand(R_t Rand) {
    double T1;
    uint64_t a = 6364136223846793005;
    unsigned b = 1;

    double dif = 0;
    double avg = (MAX + MIN)/2;

    printf("%10s\t%10s\t%10s\t%10s\t%10s\n", "Threads", "Result", "Avg", "Difference", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        experiment_result R = run_experiment_random(Rand);
        if (T == 1) {
            T1 = R.time_ms;
        }
        dif = avg - R.result;
        printf("%10u\t%10g\t%10g\t%10g\t%10g\n", T, R.result, avg, dif, T1/R.time_ms);
    };
}

//---Randomize----------------------------------------------------------------------------------------------------------

double randomize_arr_single(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    uint64_t prev = SEED;
    uint64_t sum = 0;

    for (unsigned i=0; i<n; i++){
        uint64_t cur = a*prev + b;
        V[i] = (cur % (MAX - MIN + 1)) + MIN;
        prev = cur;
        sum +=V[i];
    }

    return (double)sum/(double)n;
}

uint64_t* getLUTA(unsigned size, uint64_t a){
    uint64_t res[size+1];
    res[0] = 1;
    for (unsigned i=1; i<=size; i++) res[i] = res[i-1] * a;
    return res;
}

uint64_t* getLUTB(unsigned size, uint64_t* a, uint64_t b){
    uint64_t res[size];
    res[0] = b;
    for (unsigned i=1; i<size; i++){
        uint64_t acc = 0;
        for (unsigned j=0; j<=i; j++){
            acc += a[j];
        }
        res[i] = acc*b;
    }
    return res;
}

uint64_t getA(unsigned size, uint64_t a){
    uint64_t res = 1;
    for (unsigned i=1; i<=size; i++) res = res * a;
    return res;
}

uint64_t getB(unsigned size, uint64_t a){
    uint64_t* acc = new uint64_t(size);
    uint64_t res = 1;
    acc[0] = 1;
    for (unsigned i=1; i<=size; i++){
        for (unsigned j=0; j<i; j++){
            acc[i] = acc[j] * a;
        }
        res += acc[i];
    }
    free(acc);
    return res;
}

double randomize_arr_fs(unsigned* V, size_t n){
    uint64_t a = 6364136223846793005;
    unsigned b = 1;
    unsigned T;
//    uint64_t* LUTA;
//    uint64_t* LUTB;
    uint64_t LUTA;
    uint64_t LUTB;
    uint64_t sum = 0;

#pragma omp parallel shared(V, T, LUTA, LUTB)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) get_num_threads();
//            LUTA = getLUTA(n, a);
//            LUTB = getLUTB(n, LUTA, b);
            LUTA = getA(T, a);
            LUTB = getB((T - 1), a)*b;
        }
        uint64_t prev = SEED;
        uint64_t cur;

        for (unsigned i=t; i<n; i += T){
            if (i == t){
                cur = getA(i+1, a)*prev + getB(i, a) * b;
            } else {
                cur = LUTA*prev + LUTB;
            }
//            cur = LUTA[i+1]*prev + LUTB[i];
            V[i] = (cur % (MAX - MIN + 1)) + MIN;
            prev = cur;
        }
    }

    for (unsigned i=0; i<n;i++)
        sum += V[i];

    return (double)sum/(double)n;
}

//---Fibonacci----------------------------------------------------------------------------------------------------------

unsigned Fibonacci(unsigned n){
    if (n <= 2)
        return 1;
    return Fibonacci(n-1) + Fibonacci(n-2);
}

unsigned Fibonacci_omp(unsigned n){
    if (n <= 2)
        return 1;
    unsigned x1, x2;
#pragma omp task
    {
        x1 = Fibonacci_omp(n-1);
    };
#pragma omp task
    {
        x2 = Fibonacci_omp(n-2);
    };
#pragma omp taskwait
    return x1 + x2;
}

#include <future>
std::future<unsigned> async_Fibonacci(unsigned n)
{
    if (n <= 2) {
        auto fut = std::async([=]() { return (unsigned)1; });
        return fut;
    }
    auto fut = std::async([=]() {
        std::future<unsigned> a = async_Fibonacci(n - 1);
        std::future<unsigned> b = async_Fibonacci(n - 2);
        unsigned c = a.get() + b.get();
        return c;
    });
    return fut;
}

unsigned Fibonacci_sch_omp(unsigned n){
    unsigned acc [n];
#pragma omp for schedule(dynamic)
    for (int i=0; i<n; i++){
        if (i<=1) { acc[i] = 1; }
        else{
            acc[i] = acc[i-1] + acc[i-2];
        }
    }
    unsigned res = 0;
    for (int i=0; i<n; i++){
        res += acc[i];
    }
    return res;
}

experiment_result run_experiment_fib(F_t f) {
    double t0 = omp_get_wtime();
    double R = f(10);
    double t1 = omp_get_wtime();
    return {R, t1 - t0};
}

void show_experiment_result_Fib(F_t f) {
    double T1;
    printf("%10s\t%10s\t%10sms\t%13s\n", "Threads", "Result", "Time", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        set_num_threads(T);
        R = run_experiment_fib(f);
        if (T == 1) {
            T1 = R.time_ms;
        }
        printf("%10u\t%10g\t%10g\t%10g\n", T, R.result, R.time_ms, T1/R.time_ms);
    };
}

int main() {
    experiment_result p;

//    printf("Fibonacci omp\n");
//    show_experiment_result_Fib(Fibonacci_omp);
//    printf("Fibonacci schedule omp\n");
//    show_experiment_result_Fib(Fibonacci_sch_omp);
    printf("Fibonacci\n");
    show_experiment_result_Fib(Fibonacci);


    printf("Randomize omp fs\n");
    show_experiment_result_Rand(randomize_arr_fs);
    printf("Randomize single\n");
    show_experiment_result_Rand(randomize_arr_single);

    size_t len = 20;
    unsigned arr[len];

    cout << randomize_arr_single(arr, len) << endl;
    cout << randomize_arr_fs(arr, len) << endl;

    return 0;
}
