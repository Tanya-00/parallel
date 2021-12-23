#define STEPS 100000000
#define CACHE_LINE 64u
#ifndef PARALLEL_BARRIER_H
#define PARALLEL_BARRIER_H

#include <thread>
#include <mutex>
#include <vector>
#include <omp.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include "iostream"
#include <condition_variable>

using namespace std;

class barrier {
    bool lock_oddity = false;
    unsigned T;
    const unsigned Tmax;
    condition_variable cv;
    mutex mtx;
public:
    barrier(unsigned T);
    void arrive_and_wait();
};

#endif

barrier::barrier(unsigned T) : Tmax(T) {
    this->T = T;
};

void barrier::arrive_and_wait(){
    unique_lock lock(mtx);
    if(--T == 0){
        lock_oddity = !lock_oddity;
        T = Tmax;
        cv.notify_all();
    } else {
        auto my_lock = lock_oddity;
        while (my_lock == lock_oddity)
            cv.wait(lock);
    }
};

typedef double (*f_t)(double);
typedef double (*I_t)(f_t, double, double);
typedef struct experiment_result {
    double result;
    double time_ms;
} experiment_result;

typedef struct partial_sum_t_ {
    alignas(64) double value;
} partial_sum_t_;

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

void show_experiment_result(I_t I) {
    double T1;
    printf("%10s\t%10s\t%10sms\t%13s\n", "Threads", "Result", "Time", "Acceleration");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        omp_set_num_threads(T);
        R = run_experiment(I);
        if (T == 1) {
            T1 = R.time_ms;
        }
        printf("%10u\t%10g\t%10g\t%10g\n", T, R.result, R.time_ms, T1/R.time_ms);
    };
}

void show_experiment_result_json(I_t I, string name) {
    printf("{\n\"name\": \"%s\",\n", name.c_str());
    printf("\"points\": [\n");
    for (unsigned T = 1; T <= omp_get_num_procs(); ++T) {
        experiment_result R;
        omp_set_num_threads(T);
        R = run_experiment(I);
        printf("{ \"x\": %8u, \"y\": %8g}", T, R.time_ms);
        if (T < omp_get_num_procs()) printf(",\n");
    }
    printf("]\n}");
}

double Integrate(f_t f, double a, double b) {//integrateArr
    unsigned T;
    double Result = 0;
    double dx = (b - a) / STEPS;
    double *Accum;
#pragma omp parallel shared(Accum, T)
    {
        unsigned int t = (unsigned int) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            Accum = (double *) calloc(T, sizeof(double));
        }
        for (unsigned i = t; i < STEPS; i += T)
            Accum[t] += f(dx * i + a);
    }
    for (unsigned i = 0; i < T; ++i)
        Result += Accum[i];
    free(Accum);
    return Result * dx;
}

double integrate_cpp_mtx(f_t f, double a, double b) {
    using namespace std;
    unsigned T = get_num_threads();
    vector<thread> threads;
    mutex mtx;
    double Result = 0;
    double dx = (b - a) / STEPS;
    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &Result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T)
                R += f(dx * i + a);
            {
                scoped_lock lck{mtx};
                Result += R;
            }
        });
    }
    for (auto &thr: threads) thr.join();
    return Result * dx;
}

double integrate_crit(f_t f, double a, double b) { //IntegrateParallelOMP
    double Result = 0;
    double dx = (b - a) / STEPS;
#pragma omp parallel shared(Result)
    {
        double R = 0;
        unsigned t = omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();
        for (unsigned i = t; i < STEPS; i += T)
            R += f(i * dx + a);
#pragma omp critical
        Result += R;
    }
    return Result * dx;
}

double integrate_reduction(f_t f, double a, double b)
{
    double Result = 0;
    double dx = (b-a)/STEPS;

#pragma omp parallel for reduction(+:Result)
    for(int i = 0; i < STEPS; i++)
        Result += f(dx*i + a);

    Result *= dx;
    return Result;
}

double integrate_ps_align_omp(f_t f, double a, double b) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    partial_sum_t_ *accum = 0;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (partial_sum_t_ *) aligned_alloc(CACHE_LINE, T * sizeof(partial_sum_t_));
            memset(accum, 0, T*sizeof(*accum));
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t].value += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i].value;
    }

    free(accum);

    return result * dx;
}

double integrate_ps_cpp(f_t f, double a, double b) {
    using namespace std;
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = thread::hardware_concurrency();
    auto vec = vector(T, partial_sum_t_{0.0});
    vector<thread> thread_vec;
    auto thread_proc = [=, &vec](auto t) {
        for (unsigned i = t; i < STEPS; i += T)
            vec[t].value += f(dx * i + a);
    };
    for (unsigned t = 1; t < T; t++) {
        thread_vec.emplace_back(thread_proc, t);
    }
    thread_proc(0);
    for (auto &thread: thread_vec) {
        thread.join();
    }
    for (auto elem: vec) {
        result += elem.value;
    }
    return result * dx;
}

double integrate_cpp_atomic(f_t f, double a, double b) {
    vector<thread> threads;
    int T = get_num_threads();
    atomic<double> Result{0.0};
    double dx = (b - a) / STEPS;
    auto fun = [dx, &Result, a, b, f, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T)
            R += f(dx * i + a);
        Result = Result + R;
    };
    for (unsigned int t = 1; t < T; t++) {
        threads.emplace_back(fun, t);
    }
    fun(0);
    for (auto &thr: threads) thr.join();
    return Result * dx;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            vector<reduction_partial_result_t>(thread::hardware_concurrency(),
                                                            reduction_partial_result_t{zero});
    constexpr size_t k = 2;
    barrier bar {T};

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        size_t Mt = K / T;
        size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(size_t i = it1; i < it2; i++)
            accum = f(accum, V[i]);

        reduction_partial_results[t].value = accum;

#if 0
        size_t s = 1;
        while(s < T)
        {
            bar.arrive_and_wait();
            if((t % (s * k)) && (t + s < T))
            {
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
                s *= k;
            }
        }
#else
        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
        }
#endif
    };

    vector<thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
#if 0
requires {
    is_invocable_r_v<UnaryFn, ElementType, ElementType> &&
    is_invocable_r_v<BinaryFn, ElementType, ElementType, ElementType>
}
#endif
ElementType reduce_range(ElementType a, ElementType b, size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            vector<reduction_partial_result_t>(thread::hardware_concurrency(), reduction_partial_result_t{zero});

    barrier bar{T};
    constexpr size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        size_t Mt = K / T;
        size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i*dx));

        reduction_partial_results[t].value = accum;

        for(size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                                                              reduction_partial_results[t + s].value);
        }
    };

    vector<thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();
    return reduction_partial_results[0].value;
}

double integrate_reduction(double a, double b, f_t f){
    return reduce_range(a, b, STEPS, f, [](auto x, auto y) {return x + y;}, 0.0) * ((b - a) / STEPS);
}

int main() {
    experiment_result p;

   printf("Integrate with single(omp)\n");
    show_experiment_result(Integrate);
    printf("Integrate with critical sections(omp)\n");
    show_experiment_result(integrate_crit);
    printf("Integrate with mutex(cpp)\n");
    show_experiment_result(integrate_cpp_mtx);
   printf("Integrate reduction (omp)\n");
    show_experiment_result(integrate_reduction);
    printf("Integrate aligned array with partial sums(omp)\n");
    show_experiment_result(integrate_ps_align_omp);
    printf("Integrate with partial sums(cpp)\n");
    show_experiment_result(integrate_ps_cpp);
    printf("Integrate with atomic operations(cpp)\n");
    show_experiment_result(integrate_cpp_atomic);
    printf("Integrate with barrier(cpp)\n");
    show_experiment_result(integrate_reduction);
//    printf("{\n\"series\": [\n");
//    show_experiment_result_json(Integrate, "Integrate");
//    printf(",");
//    show_experiment_result_json(integrate_crit, "integrate_crit");
//    printf(",");
//    show_experiment_result_json(integrate_cpp_mtx, "integrate_cpp_mtx");
//    printf(",");
//    show_experiment_result_json(integrate_ps, "integrate_ps_cpp");
//    printf(",");
//    show_experiment_result_json(integratePS, "integratePS");
//    printf("]}");
//    show_experiment_result_json(integrate_cpp_atomic, "integrate_cpp_atomic");
//    printf("]}");

    return 0;
}