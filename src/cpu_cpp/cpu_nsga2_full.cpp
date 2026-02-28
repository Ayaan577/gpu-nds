/*
 * cpu_nsga2_full.cpp
 * Complete, self-contained C++ NSGA-II implementation.
 * All operators (NDS, crowding distance, tournament selection,
 * SBX crossover, polynomial mutation, objective evaluation)
 * are implemented in C++ — no Python or NumPy in the loop.
 *
 * Compile (Windows):
 *   g++ -O3 -march=native -ffast-math -std=c++17 ^
 *       -shared -static-libgcc -static-libstdc++ ^
 *       -o cpu_nsga2_full.dll cpu_nsga2_full.cpp
 *
 * Compile (Linux):
 *   g++ -O3 -march=native -ffast-math -std=c++17 ^
 *       -shared -fPIC -o libcpu_nsga2_full.so cpu_nsga2_full.cpp
 */

#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <numeric>
#include <functional>
#include <cstring>
#include <cfloat>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================
 * Dominance check: does solution i dominate solution j?
 * ================================================================ */
static inline bool dominates(const float* F, int i, int j, int M) {
    bool dominated = true;
    bool strict = false;
    for (int m = 0; m < M; m++) {
        float a = F[i * M + m];
        float b = F[j * M + m];
        if (a > b) { dominated = false; break; }
        if (a < b) strict = true;
    }
    return dominated && strict;
}

/* ================================================================
 * Non-Dominated Sort (NSGA-II style, O(MN^2))
 * ================================================================ */
static void nds(const float* F, int N, int M, int* ranks) {
    std::vector<int> dom_count(N, 0);
    std::vector<std::vector<int>> dom_set(N);

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (dominates(F, i, j, M)) {
                dom_set[i].push_back(j);
                dom_count[j]++;
            } else if (dominates(F, j, i, M)) {
                dom_set[j].push_back(i);
                dom_count[i]++;
            }
        }
    }

    // Iterative front peeling
    std::vector<int> current_front;
    current_front.reserve(N);
    for (int i = 0; i < N; i++) {
        ranks[i] = -1;
        if (dom_count[i] == 0) current_front.push_back(i);
    }

    int front = 0;
    while (!current_front.empty()) {
        std::vector<int> next_front;
        for (int i : current_front) {
            ranks[i] = front;
            for (int j : dom_set[i]) {
                dom_count[j]--;
                if (dom_count[j] == 0) next_front.push_back(j);
            }
        }
        current_front = std::move(next_front);
        front++;
    }
}

/* ================================================================
 * Crowding Distance
 * ================================================================ */
static void crowding_distance(const float* F, const int* ranks, float* cd,
                              int N, int M) {
    for (int i = 0; i < N; i++) cd[i] = 0.0f;

    int max_rank = 0;
    for (int i = 0; i < N; i++) {
        if (ranks[i] > max_rank) max_rank = ranks[i];
    }

    std::vector<int> front_idx;
    front_idx.reserve(N);

    for (int k = 0; k <= max_rank; k++) {
        front_idx.clear();
        for (int i = 0; i < N; i++) {
            if (ranks[i] == k) front_idx.push_back(i);
        }
        int nf = (int)front_idx.size();
        if (nf <= 2) {
            for (int i : front_idx) cd[i] = FLT_MAX;
            continue;
        }

        for (int m = 0; m < M; m++) {
            // Sort front indices by objective m
            std::sort(front_idx.begin(), front_idx.end(),
                [&](int a, int b) {
                    return F[a * M + m] < F[b * M + m];
                });

            float f_min = F[front_idx[0] * M + m];
            float f_max = F[front_idx[nf - 1] * M + m];
            float f_range = f_max - f_min;
            if (f_range < 1e-12f) continue;

            cd[front_idx[0]] = FLT_MAX;
            cd[front_idx[nf - 1]] = FLT_MAX;

            for (int i = 1; i < nf - 1; i++) {
                float dist = (F[front_idx[i + 1] * M + m] -
                              F[front_idx[i - 1] * M + m]) / f_range;
                cd[front_idx[i]] += dist;
            }
        }
    }
}

/* ================================================================
 * Tournament Selection (binary)
 * ================================================================ */
static void tournament_select(const int* ranks, const float* cd, int N,
                              int* selected, int n_select,
                              std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, N - 1);
    for (int i = 0; i < n_select; i++) {
        int a = dist(rng);
        int b = dist(rng);
        while (b == a) b = dist(rng);

        if (ranks[a] < ranks[b]) {
            selected[i] = a;
        } else if (ranks[b] < ranks[a]) {
            selected[i] = b;
        } else if (cd[a] >= cd[b]) {
            selected[i] = a;
        } else {
            selected[i] = b;
        }
    }
}

/* ================================================================
 * SBX Crossover
 * ================================================================ */
static void sbx_crossover(const float* X, const int* parent_idx, int N,
                           int n_var, const float* xl, const float* xu,
                           float* X_off, float eta_c,
                           std::mt19937& rng) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    int n_pairs = N / 2;

    for (int i = 0; i < n_pairs; i++) {
        int p1 = parent_idx[i];
        int p2 = parent_idx[i + n_pairs];

        for (int j = 0; j < n_var; j++) {
            float x1 = X[p1 * n_var + j];
            float x2 = X[p2 * n_var + j];
            float u = u01(rng);

            float beta;
            if (u <= 0.5f) {
                beta = std::pow(2.0f * u, 1.0f / (eta_c + 1.0f));
            } else {
                beta = std::pow(1.0f / (2.0f * (1.0f - u)),
                                1.0f / (eta_c + 1.0f));
            }

            float c1 = 0.5f * ((1.0f + beta) * x1 + (1.0f - beta) * x2);
            float c2 = 0.5f * ((1.0f - beta) * x1 + (1.0f + beta) * x2);

            // Clip to bounds
            c1 = std::max(xl[j], std::min(xu[j], c1));
            c2 = std::max(xl[j], std::min(xu[j], c2));

            X_off[i * n_var + j] = c1;
            X_off[(i + n_pairs) * n_var + j] = c2;
        }
    }
}

/* ================================================================
 * Polynomial Mutation
 * ================================================================ */
static void poly_mutation(float* X, int N, int n_var,
                          const float* xl, const float* xu,
                          float eta_m, float prob_m,
                          std::mt19937& rng) {
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_var; j++) {
            if (u01(rng) < prob_m) {
                float x = X[i * n_var + j];
                float lb = xl[j];
                float ub = xu[j];
                float range = ub - lb;
                if (range < 1e-10f) continue;

                float delta1 = (x - lb) / range;
                float delta2 = (ub - x) / range;
                float u = u01(rng);
                float dq;

                if (u < 0.5f) {
                    float val = 2.0f * u +
                        (1.0f - 2.0f * u) *
                        std::pow(1.0f - delta1, eta_m + 1.0f);
                    dq = std::pow(val, 1.0f / (eta_m + 1.0f)) - 1.0f;
                } else {
                    float val = 2.0f * (1.0f - u) +
                        2.0f * (u - 0.5f) *
                        std::pow(1.0f - delta2, eta_m + 1.0f);
                    dq = 1.0f - std::pow(val, 1.0f / (eta_m + 1.0f));
                }

                X[i * n_var + j] = std::max(lb,
                    std::min(ub, x + dq * range));
            }
        }
    }
}

/* ================================================================
 * Objective Evaluation Functions
 * ================================================================ */
static void eval_zdt1(const float* X, float* F, int N, int n_var) {
    for (int i = 0; i < N; i++) {
        float f1 = X[i * n_var + 0];
        float sum = 0.0f;
        for (int j = 1; j < n_var; j++) sum += X[i * n_var + j];
        float g = 1.0f + 9.0f * sum / (float)(n_var - 1);
        float f2 = g * (1.0f - std::sqrt(f1 / g));
        F[i * 2 + 0] = f1;
        F[i * 2 + 1] = f2;
    }
}

static void eval_zdt2(const float* X, float* F, int N, int n_var) {
    for (int i = 0; i < N; i++) {
        float f1 = X[i * n_var + 0];
        float sum = 0.0f;
        for (int j = 1; j < n_var; j++) sum += X[i * n_var + j];
        float g = 1.0f + 9.0f * sum / (float)(n_var - 1);
        float f2 = g * (1.0f - (f1 / g) * (f1 / g));
        F[i * 2 + 0] = f1;
        F[i * 2 + 1] = f2;
    }
}

static void eval_zdt3(const float* X, float* F, int N, int n_var) {
    for (int i = 0; i < N; i++) {
        float f1 = X[i * n_var + 0];
        float sum = 0.0f;
        for (int j = 1; j < n_var; j++) sum += X[i * n_var + j];
        float g = 1.0f + 9.0f * sum / (float)(n_var - 1);
        float f2 = g * (1.0f - std::sqrt(f1 / g) -
                        (f1 / g) * std::sin(10.0f * (float)M_PI * f1));
        F[i * 2 + 0] = f1;
        F[i * 2 + 1] = f2;
    }
}

static void eval_dtlz2(const float* X, float* F, int N,
                        int n_var, int n_obj) {
    int k = n_var - n_obj + 1;
    for (int i = 0; i < N; i++) {
        // Compute g
        float g = 0.0f;
        for (int j = n_obj - 1; j < n_var; j++) {
            float diff = X[i * n_var + j] - 0.5f;
            g += diff * diff;
        }

        for (int m = 0; m < n_obj; m++) {
            float fm = 1.0f + g;
            for (int j = 0; j < n_obj - 1 - m; j++) {
                fm *= std::cos(X[i * n_var + j] * (float)M_PI / 2.0f);
            }
            if (m > 0) {
                fm *= std::sin(X[i * n_var + (n_obj - 1 - m)] *
                               (float)M_PI / 2.0f);
            }
            F[i * n_obj + m] = fm;
        }
    }
}

static void evaluate(const float* X, float* F, int N,
                     int n_var, int n_obj, int problem_id) {
    switch (problem_id) {
        case 0: eval_zdt1(X, F, N, n_var); break;
        case 1: eval_zdt2(X, F, N, n_var); break;
        case 2: eval_zdt3(X, F, N, n_var); break;
        case 3: eval_dtlz2(X, F, N, n_var, 3); break;
        case 4: eval_dtlz2(X, F, N, n_var, 5); break;
        default: eval_dtlz2(X, F, N, n_var, n_obj); break;
    }
}

/* ================================================================
 * Environmental Selection (truncation)
 * Select top N by (rank ASC, crowding distance DESC)
 * ================================================================ */
static void environmental_selection(
    const float* X_comb, const float* F_comb,
    const int* ranks, const float* cd,
    int N2, int N_target, int n_var, int n_obj,
    float* X_out, float* F_out)
{
    std::vector<int> idx(N2);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (ranks[a] != ranks[b]) return ranks[a] < ranks[b];
        return cd[a] > cd[b];  // higher CD preferred
    });

    for (int i = 0; i < N_target; i++) {
        int src = idx[i];
        std::memcpy(X_out + i * n_var, X_comb + src * n_var,
                    n_var * sizeof(float));
        std::memcpy(F_out + i * n_obj, F_comb + src * n_obj,
                    n_obj * sizeof(float));
    }
}

/* ================================================================
 * Main NSGA-II loop
 * Returns total elapsed time in milliseconds (excluding init).
 * ================================================================ */
extern "C"
#ifdef _WIN32
__declspec(dllexport)
#endif
double cpu_nsga2_run(
    int N,          // population size
    int n_var,      // number of decision variables
    int n_obj,      // number of objectives
    int n_gen,      // number of generations
    float* xl,      // lower bounds (n_var,)
    float* xu,      // upper bounds (n_var,)
    int seed,       // random seed
    float* F_out,   // output: final objectives (N, n_obj)
    float* X_out,   // output: final decisions (N, n_var)
    int problem_id  // 0=ZDT1, 1=ZDT2, 2=ZDT3, 3=DTLZ2_M3, 4=DTLZ2_M5
)
{
    // MOEA parameters
    float eta_c = 20.0f;
    float eta_m = 20.0f;
    float prob_m = 1.0f / (float)n_var;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);

    // Allocate all arrays
    std::vector<float> X(N * n_var);
    std::vector<float> F(N * n_obj);
    std::vector<float> X_off(N * n_var);
    std::vector<float> F_off(N * n_obj);
    std::vector<float> X_comb(2 * N * n_var);
    std::vector<float> F_comb(2 * N * n_obj);
    std::vector<int> ranks(2 * N);
    std::vector<float> cd(2 * N);
    std::vector<int> parent_idx(N);

    // Initialize population randomly in [xl, xu]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_var; j++) {
            X[i * n_var + j] = xl[j] + u01(rng) * (xu[j] - xl[j]);
        }
    }

    // Initial evaluation
    evaluate(X.data(), F.data(), N, n_var, n_obj, problem_id);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int gen = 0; gen < n_gen; gen++) {
        // 1. NDS on current pop
        nds(F.data(), N, n_obj, ranks.data());

        // 2. Crowding distance
        crowding_distance(F.data(), ranks.data(), cd.data(), N, n_obj);

        // 3. Tournament selection
        tournament_select(ranks.data(), cd.data(), N,
                          parent_idx.data(), N, rng);

        // 4. SBX crossover
        sbx_crossover(X.data(), parent_idx.data(), N, n_var,
                       xl, xu, X_off.data(), eta_c, rng);

        // 5. Polynomial mutation
        poly_mutation(X_off.data(), N, n_var, xl, xu, eta_m, prob_m, rng);

        // 6. Evaluate offspring
        evaluate(X_off.data(), F_off.data(), N, n_var, n_obj, problem_id);

        // 7. Combine: X_comb = [X; X_off], F_comb = [F; F_off]
        std::memcpy(X_comb.data(), X.data(),
                    N * n_var * sizeof(float));
        std::memcpy(X_comb.data() + N * n_var, X_off.data(),
                    N * n_var * sizeof(float));
        std::memcpy(F_comb.data(), F.data(),
                    N * n_obj * sizeof(float));
        std::memcpy(F_comb.data() + N * n_obj, F_off.data(),
                    N * n_obj * sizeof(float));

        // 8. NDS on combined population (2N)
        nds(F_comb.data(), 2 * N, n_obj, ranks.data());

        // 9. Crowding distance on combined
        crowding_distance(F_comb.data(), ranks.data(), cd.data(),
                          2 * N, n_obj);

        // 10. Environmental selection: top N by (rank, -cd)
        environmental_selection(
            X_comb.data(), F_comb.data(),
            ranks.data(), cd.data(),
            2 * N, N, n_var, n_obj,
            X.data(), F.data());
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Copy final results to output
    std::memcpy(F_out, F.data(), N * n_obj * sizeof(float));
    std::memcpy(X_out, X.data(), N * n_var * sizeof(float));

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
