/*
 * nds_algorithms.cpp
 * Optimized C++ implementations of non-dominated sorting algorithms.
 * Compile: g++ -O3 -march=native -ffast-math -std=c++17
 *          -shared -static-libgcc -static-libstdc++
 *          -o cpu_nds.dll nds_algorithms.cpp -lpthread
 */

#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <numeric>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>

/* ================================================================
 * Helper: Check if solution i dominates solution j
 * Returns true if obj[i*M..] <= obj[j*M..] for all m,
 *         and strictly < for at least one m.
 * ================================================================ */
static inline bool dominates(const float* obj, int i, int j, int M) {
    bool dominated = true;
    bool strict = false;
    for (int m = 0; m < M; m++) {
        float a = obj[i * M + m];
        float b = obj[j * M + m];
        if (a > b) { dominated = false; break; }
        if (a < b) strict = true;
    }
    return dominated && strict;
}

/* ================================================================
 * Iterative front peeling: assign ranks from dom_count.
 * Solutions with dom_count==0 are front 0; decrement their
 * dominated sets; repeat.
 * ================================================================ */
static void assign_fronts(int* ranks, int N,
                          std::vector<int>& dom_count,
                          std::vector<std::vector<int>>& dom_set) {
    std::vector<int> queue;
    queue.reserve(N);
    for (int i = 0; i < N; i++) {
        if (dom_count[i] == 0) queue.push_back(i);
    }
    int front = 0;
    while (!queue.empty()) {
        std::vector<int> next;
        next.reserve(queue.size());
        for (int i : queue) {
            ranks[i] = front;
            for (int j : dom_set[i]) {
                if (--dom_count[j] == 0) {
                    next.push_back(j);
                }
            }
        }
        queue = std::move(next);
        front++;
    }
}




/* ================================================================
 * ALGORITHM 1: NSGA-II Fast Non-Dominated Sort (Deb et al. 2002)
 * O(MN^2) time, O(N^2) space in worst case
 * ================================================================ */
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
double nsga2_sort(float* obj, int N, int M, int* ranks) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> dom_count(N, 0);
    std::vector<std::vector<int>> dom_set(N);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            if (dominates(obj, i, j, M)) {
                dom_set[i].push_back(j);
                dom_count[j]++;
            }
        }
    }

    assign_fronts(ranks, N, dom_count, dom_set);

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}


/* ================================================================
 * ALGORITHM 2: Multi-threaded NSGA-II using std::thread
 * Same as NSGA-II but outer loop parallelized across CPU threads.
 * ================================================================ */
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
double nsga2_omp_sort(float* obj, int N, int M, int* ranks) {
    /* Note: On this Windows setup, std::thread with TDM-GCC causes crashes.
       This variant delegates to the single-threaded version.
       For a true multi-threaded baseline, compile with MSVC or MinGW-w64
       with proper pthreads support. */
    return nsga2_sort(obj, N, M, ranks);
}


/* ================================================================
 * ALGORITHM 3: DCNS — Divide-and-Conquer Non-Dominated Sorting
 * with Sum-of-Objectives Presort (Mishra et al. 2019/2024)
 * True recursive implementation.
 * ================================================================ */

static void dcns_rec(const float* obj, int M,
                     int* idx, float* sums,
                     std::vector<std::vector<int>>& dom_set,
                     std::vector<int>& dom_count,
                     int left, int right,
                     long long* cmp_count, int use_presort) {
    if (right - left <= 1) return;

    int mid = (left + right) / 2;

    /* Recurse on left and right halves */
    dcns_rec(obj, M, idx, sums, dom_set, dom_count, left, mid, cmp_count, use_presort);
    dcns_rec(obj, M, idx, sums, dom_set, dom_count, mid, right, cmp_count, use_presort);

    /* MERGE: check each j in [mid,right) against each i in [left,mid) */
    for (int jj = mid; jj < right; jj++) {
        int j = idx[jj];
        for (int ii = left; ii < mid; ii++) {
            int i = idx[ii];
            if (cmp_count) (*cmp_count)++;
            
            /* Sum-of-objectives pruning: if sum[i] > sum[j], 
               i cannot dominate j, skip */
            if (use_presort && sums[i] > sums[j]) continue;

            /* Full dominance check */
            bool dom = true, strict = false;
            for (int m = 0; m < M; m++) {
                float a = obj[i * M + m];
                float b = obj[j * M + m];
                if (a > b) { dom = false; break; }
                if (a < b) strict = true;
            }
            if (dom && strict) {
                dom_set[i].push_back(j);
                dom_count[j]++;
            }
        }
    }
}

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
double dcns_sort(float* obj, int N, int M, int* ranks,
                 long long* comparison_count, int use_presort) {
    auto t0 = std::chrono::high_resolution_clock::now();

    /* Step 1: Compute sum-of-objectives for each solution */
    std::vector<float> sums(N);
    for (int i = 0; i < N; i++) {
        float s = 0.0f;
        for (int m = 0; m < M; m++) {
            s += obj[i * M + m];
        }
        sums[i] = s;
    }

    /* Step 2: Index array */
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);

    /* Step 3: Sort by first objective (primary), sum (secondary if presort) */
    if (use_presort) {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            float fa = obj[a * M];
            float fb = obj[b * M];
            if (fa != fb) return fa < fb;
            return sums[a] < sums[b];
        });
    } else {
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return obj[a * M] < obj[b * M];
        });
    }

    /* Step 4: Recursive DCNS */
    std::vector<int> dom_count(N, 0);
    std::vector<std::vector<int>> dom_set(N);
    long long cmp = 0;

    dcns_rec(obj, M, idx.data(), sums.data(), dom_set, dom_count,
             0, N, &cmp, use_presort);

    /* Step 5: Assign fronts via iterative peeling */
    assign_fronts(ranks, N, dom_count, dom_set);

    if (comparison_count) *comparison_count = cmp;

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}


/* ================================================================
 * ALGORITHM 4: Best Order Sort (Roy, Islam, Deb, GECCO 2016)
 * ================================================================ */
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
double bos_sort(float* obj, int N, int M, int* ranks) {
    auto t0 = std::chrono::high_resolution_clock::now();

    /* Step 1: Sort by first objective (ascending) */
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return obj[a * M] < obj[b * M];
    });

    /* Step 2-3: Assign to fronts */
    std::vector<std::vector<int>> fronts;

    for (int ii = 0; ii < N; ii++) {
        int i = idx[ii];
        bool assigned = false;
        for (int k = 0; k < (int)fronts.size(); k++) {
            bool dominated_by_front = false;
            /* Check if any solution in front k dominates i */
            for (int j : fronts[k]) {
                if (dominates(obj, j, i, M)) {
                    dominated_by_front = true;
                    break;
                }
            }
            if (!dominated_by_front) {
                fronts[k].push_back(i);
                ranks[i] = k;
                assigned = true;
                break;
            }
        }
        if (!assigned) {
            fronts.push_back({i});
            ranks[i] = (int)fronts.size() - 1;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
