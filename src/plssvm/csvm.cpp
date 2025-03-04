/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/csvm.hpp"

#include "plssvm/backends/SYCL/AdaptiveCpp/detail/device_ptr.hpp"  // TODO
#include "plssvm/constants.hpp"                                    // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/assert.hpp"                                // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                               // plssvm::detail::log
#include "plssvm/detail/move_only_any.hpp"                         // plssvm::detail::move_only_any
#include "plssvm/detail/operators.hpp"                             // plssvm operator overloads for vectors
#include "plssvm/detail/tracking/performance_tracker.hpp"          // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT, plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/utility.hpp"                               // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"                        // plssvm::invalid_parameter_exception
#include "plssvm/gamma.hpp"                                        // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"                        // plssvm::kernel_function_type
#include "plssvm/kernel_functions.hpp"                             // plssvm::kernel_function
#include "plssvm/matrix.hpp"                                       // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                                    // plssvm::parameter
#include "plssvm/shape.hpp"                                        // plssvm::shape
#include "plssvm/solver_types.hpp"                                 // plssvm::solver_type
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity_level

#include "fmt/format.h"  // fmt::format

#include <algorithm>      // std::count
#include <chrono>         // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>        // std::size_t
#include <functional>     // std::plus
#include <numeric>        // std::inner_product
#include <sycl/sycl.hpp>  // TODO
#include <utility>        // std::move
#include <utility>        // std::pair, std::make_pair
#include <variant>        // std::holds_alternative, std::get
#include <vector>         // std::vector

namespace plssvm {

void csvm::sanity_check_parameter() const {
    // kernel: valid kernel function
    const auto kernel_type_value = detail::to_underlying(params_.kernel_type);
    if (kernel_type_value < 0 || kernel_type_value >= 6) {
        throw invalid_parameter_exception{ fmt::format("Invalid kernel function with value {} given!", kernel_type_value) };
    }

    // gamma: must be greater than 0 IF explicitly provided as real_type (not for the linear kernel)
    if (params_.kernel_type != kernel_function_type::linear && std::holds_alternative<real_type>(params_.gamma) && std::get<real_type>(params_.gamma) <= real_type{ 0.0 }) {
        throw invalid_parameter_exception{ fmt::format("gamma must be greater than 0.0, but is {}!", std::get<real_type>(params_.gamma)) };
    }
    // degree: all allowed
    // coef0: all allowed
    // cost: all allowed
}

std::pair<soa_matrix<real_type>, std::vector<unsigned long long>> csvm::conjugate_gradients(const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, const real_type eps, const unsigned long long max_cg_iter, const solver_type cg_solver) const {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(!B.empty(), "The right-hand sides must not be empty!");
    PLSSVM_ASSERT(eps > real_type{ 0.0 }, "The epsilon value must be greater than 0.0!");
    PLSSVM_ASSERT(max_cg_iter > 0, "The maximum number of iterations must be greater than 0!");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("cg start");

    const std::size_t num_rows = B.num_cols();
    const std::size_t num_rhs = B.num_rows();

    // timing for each CG iteration
    std::chrono::milliseconds total_iteration_time{};
    std::vector<std::chrono::milliseconds> blas_level_3_times{};

    // track the number of iterations needed per rhs
    unsigned long long iter = 0;
    std::vector<unsigned long long> num_iters(num_rhs, 1);

    //
    // perform Conjugate Gradients (CG) algorithm
    //

    soa_matrix<real_type> X{ shape{ num_rhs, num_rows }, real_type{ 1.0 }, shape{ PADDING_SIZE, PADDING_SIZE } };

    // R = B - A * X
    soa_matrix<real_type> R{ B, shape{ PADDING_SIZE, PADDING_SIZE } };
    blas_level_3_times.push_back(this->run_blas_level_3(cg_solver, real_type{ -1.0 }, A, X, real_type{ 1.0 }, R));

    // delta = R.T * R
    std::vector<real_type> delta = rowwise_dot(R, R);
    const std::vector<real_type> delta0(delta);

    soa_matrix<real_type> D{ R, shape{ PADDING_SIZE, PADDING_SIZE } };

    // get the index of the rhs that has the largest residual difference wrt to its target residual
    const auto rhs_idx_max_residual_difference = [&]() {
        const real_type max_difference{ 0.0 };
        std::size_t idx{ 0 };
        for (std::size_t i = 0; i < delta.size(); ++i) {
            const real_type difference = delta[i] - (eps * eps * delta0[i]);
            if (difference > max_difference) {
                idx = i;
            }
        }
        return idx;
    };

    std::vector<unsigned long long> mask(num_rhs, 1);
    // calculate a mask for every converged right hand side
    // -> 0 if the rhs already converged, 1 otherwise
    const auto calculate_rhs_converged_mask = [eps, delta0, &mask](const std::vector<real_type> &delta_vec, const soa_matrix<real_type> &R_matr) {
#pragma omp parallel for shared(delta_vec, R_matr)
        for (std::size_t row = 0; row < R_matr.num_rows(); ++row) {
            // check if this rhs is already marked as converged
            if (mask[row] == 1) {
                // check if this rhs is now converged
                if (delta_vec[row] <= eps * eps * delta0[row]) {
                    // the residual of this rhs is already small enough -> converged
                    mask[row] = 0;
                } else {
                    // the residual of this rhs is all zeros -> converged
                    bool is_residual_zero = true;
                    for (std::size_t col = 0; col < R_matr.num_cols(); ++col) {
                        if (R_matr(row, col) != real_type{ 0.0 }) {
                            is_residual_zero = false;
                            break;
                        }
                    }
                    // all residual values are 0 -> residual is 0 -> can't updated X for this rhs!
                    if (is_residual_zero) {
                        mask[row] = 0;
                    }
                }
            }
        }
        return mask;
    };
    // get the number of rhs that have already been converged
    const auto num_rhs_converged = [&mask]() {
        return static_cast<std::size_t>(std::count(mask.cbegin(), mask.cend(), 0));
    };

    while (iter < max_cg_iter && num_rhs_converged() < num_rhs) {
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT(fmt::format("cg iter {} start", iter));

        const std::size_t max_residual_difference_idx = rhs_idx_max_residual_difference();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with {}/{} converged rhs (max residual {} with target residual {} for rhs {}). ",
                    iter + 1,
                    max_cg_iter,
                    num_rhs_converged(),
                    num_rhs,
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        const std::chrono::steady_clock::time_point iteration_start_time = std::chrono::steady_clock::now();

        // create mask for the residual -> only update X if the respective rhs did not already converge
        mask = calculate_rhs_converged_mask(delta, R);

        // Q = A * D
        soa_matrix<real_type> Q{ shape{ D.num_rows(), D.num_cols() }, shape{ PADDING_SIZE, PADDING_SIZE } };
        blas_level_3_times.push_back(this->run_blas_level_3(cg_solver, real_type{ 1.0 }, A, D, real_type{ 0.0 }, Q));

        // alpha = delta_new / (D^T * Q))
        const std::vector<real_type> alpha = delta / rowwise_dot(D, Q);

        // X = X + alpha * D
        X += masked_rowwise_scale(mask, alpha, D);

        if (iter % 50 == 49) {
            // explicitly recalculate residual to remove accumulating floating point errors
            // R = B - A * X
            R = soa_matrix<real_type>{ B, shape{ PADDING_SIZE, PADDING_SIZE } };
            blas_level_3_times.push_back(this->run_blas_level_3(cg_solver, real_type{ -1.0 }, A, X, real_type{ 1.0 }, R));
        } else {
            // R = R - alpha * Q
            R -= rowwise_scale(alpha, Q);
        }

        // delta = R^T * R
        const std::vector<real_type> delta_old = delta;
        delta = rowwise_dot(R, R);

        // beta = delta_new / delta_old
        const std::vector<real_type> beta = delta / delta_old;
        // D = beta * D + R
        D = rowwise_scale(beta, D) + R;

        const std::chrono::steady_clock::time_point iteration_end_time = std::chrono::steady_clock::now();
        const std::chrono::duration iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n",
                    iteration_duration);
        total_iteration_time += iteration_duration;

        // next CG iteration
        ++iter;
        num_iters += mask;
    }
    const std::size_t max_residual_difference_idx = rhs_idx_max_residual_difference();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {}.\n",
                detail::tracking::tracking_entry{ "cg", "iterations", iter },
                detail::tracking::tracking_entry{ "cg", "max_iterations", max_cg_iter },
                detail::tracking::tracking_entry{ "cg", "num_converged_rhs", num_rhs_converged() },
                detail::tracking::tracking_entry{ "cg", "num_rhs", num_rhs },
                delta[max_residual_difference_idx],
                eps * eps * delta0[max_residual_difference_idx],
                max_residual_difference_idx,
                detail::tracking::tracking_entry{ "cg", "avg_iteration_time", total_iteration_time / std::max(iter, 1ULL) });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "cg", "blas_level_3_times", blas_level_3_times }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "cg", "residuals", delta }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "cg", "target_residuals", eps * eps * delta0 }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "cg", "epsilon", eps }));
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n",
                iter);

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("cg end");

    return std::make_pair(X, num_iters);
}

// ------------------------------------
// ------------------------------------
// ------------------------------------
// Start of Cholesky Implementation
// ------------------------------------
// ------------------------------------
// ------------------------------------

//
// Constants
//

constexpr size_t WIDTH = PADDING_SIZE;  // PADDING and THREAD^2 (best for NVIDIA A30, any floating point)
                                        // Required, since each Thread access one element in a block
                                        // Best is 32 since warp size

#ifdef PLSSVM_FLOAT_AS_REAL_TYPE
constexpr size_t CHOL_THREADCOLWIDTH = 10;
constexpr size_t CHOL_THREADSUBWIDTH = 6;
constexpr size_t CHOL_THREADSUBHEIGHT = 6;
constexpr size_t FSUB_THREADCOLHEIGHT = 1;
constexpr size_t FSUB_THREAD_L_LEN = 10;
constexpr size_t FSUB_THREAD_B_LEN = 1;
constexpr size_t BSUB_THREADCOLHEIGHT = 1;
constexpr size_t BSUB_THREAD_L_LEN = 10;
constexpr size_t BSUB_THREAD_B_LEN = 1;
#else
constexpr size_t CHOL_THREADCOLWIDTH = 10;
constexpr size_t CHOL_THREADSUBWIDTH = 3;
constexpr size_t CHOL_THREADSUBHEIGHT = 3;
constexpr size_t FSUB_THREADCOLHEIGHT = 1;
constexpr size_t FSUB_THREAD_L_LEN = 5;
constexpr size_t FSUB_THREAD_B_LEN = 1;
constexpr size_t BSUB_THREADCOLHEIGHT = 1;
constexpr size_t BSUB_THREAD_L_LEN = 4;
constexpr size_t BSUB_THREAD_B_LEN = 1;
#endif

//
// Math
//

constexpr auto ceil_div(size_t a, size_t b) -> size_t {
    return a / b + (a % b != 0);
}

//
// Indexing (unoptimized, meaning probably can be shortend)
//

template <size_t WIDTH>
constexpr auto blockcol_idx(size_t N, size_t i, size_t j) -> size_t {
    if (i >= N) {
        i = N;
        j = N - 1;
    }

    return i * (2 * (N + WIDTH) - i + 1) / 2 - i + j;
}

constexpr auto blockvec_idx(size_t Naligned, size_t i, size_t j) -> size_t {
    return Naligned * i + j;
}

template <size_t WIDTH>
constexpr auto blockvec_idx_alloc(size_t Ndiv, size_t Nbdiv) -> size_t {
    return Ndiv * Nbdiv * WIDTH * WIDTH;
}

//
// Kernels
//

template <size_t WIDTH>
void cholesky_decomposition(::sycl::group<2> g,
                            ::sycl::local_accessor<real_type, 2> cholblk,
                            real_type jitter) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    for (size_t k = 0; k < WIDTH - 1; ++k) {
        if (locali > k) {
            cholblk[locali][localj] -= cholblk[k][locali] * cholblk[k][localj] / (cholblk[k][k] + jitter);
        }

        ::sycl::group_barrier(g);
    }

    const auto diag = ::sycl::sqrt(cholblk[locali][locali] + jitter);

    ::sycl::group_barrier(g);

    cholblk[locali][localj] = (locali == localj) * diag + (locali != localj) * cholblk[locali][localj] / diag;
}

template <size_t WIDTH, size_t THREADWIDTH>
void solve_forwardsub(::sycl::group<2> g,
                      ::sycl::local_accessor<real_type, 2> triagblk,
                      ::sycl::local_accessor<real_type, 2> prevrslt,
                      real_type (&x)[THREADWIDTH]) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    const auto diag = triagblk[locali][locali];

    for (size_t k = 0; k < WIDTH - 1; ++k) {
        if (locali == k) {
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                prevrslt[u][localj] = x[u] /= diag;
            }
        }

        ::sycl::group_barrier(g);

        if (locali > k) {
            const auto chol = triagblk[k][locali];
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                x[u] -= chol * prevrslt[u][localj];
            }
        }

        ::sycl::group_barrier(g);
    }

    if (locali == WIDTH - 1) {
        for (size_t u = 0; u < THREADWIDTH; ++u) {
            x[u] /= diag;
        }
    }
}

template <size_t WIDTH, size_t THREADTOPWIDTH, size_t THREADLEFTHEIGHT>
void trans_matmul(::sycl::group<2> g,
                  ::sycl::local_accessor<real_type, 3> leftblk,
                  ::sycl::local_accessor<real_type, 3> topblk,
                  real_type (&sum)[THREADLEFTHEIGHT * THREADTOPWIDTH]) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    for (size_t k = 0; k < WIDTH; ++k) {
        real_type left[THREADLEFTHEIGHT];
        for (size_t u = 0; u < THREADLEFTHEIGHT; ++u) {
            left[u] = leftblk[u][k][locali];
        }

        real_type top[THREADTOPWIDTH];
        for (size_t u = 0; u < THREADTOPWIDTH; ++u) {
            top[u] = topblk[u][k][localj];
        }

        for (size_t i = 0; i < THREADLEFTHEIGHT; ++i) {
            for (size_t j = 0; j < THREADTOPWIDTH; ++j) {
                sum[i * THREADTOPWIDTH + j] += left[i] * top[j];
            }
        }
    }
}

template <size_t WIDTH>
void transpose(::sycl::nd_item<2> idx,
               real_type *mat,
               ::sycl::local_accessor<real_type, 2> blk) {
    const auto globalij = idx.get_global_linear_id();
    const auto locali = idx.get_local_id(0);
    const auto localj = idx.get_local_id(1);

    blk[locali][localj] = mat[globalij];

    ::sycl::group_barrier(idx.get_group());

    mat[globalij] = blk[localj][locali];
}

template <size_t WIDTH, size_t THREADWIDTH>
void solve_backwardsub(::sycl::group<2> g,
                       ::sycl::local_accessor<real_type, 2> triagblk,
                       ::sycl::local_accessor<real_type, 2> prevrslt,
                       real_type (&x)[THREADWIDTH]) {
    const auto locali = g.get_local_id(0);
    const auto localj = g.get_local_id(1);

    const auto diag = triagblk[locali][locali];

    for (size_t _k = 0; _k < WIDTH - 1; ++_k) {
        const auto k = WIDTH - 1 - _k;

        if (locali == k) {
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                prevrslt[u][localj] = x[u] /= diag;
            }
        }

        ::sycl::group_barrier(g);

        if (locali < k) {
            const auto chol = triagblk[locali][k];
            for (size_t u = 0; u < THREADWIDTH; ++u) {
                x[u] -= chol * prevrslt[u][localj];
            }
        }

        ::sycl::group_barrier(g);
    }

    if (locali == 0) {
        for (size_t u = 0; u < THREADWIDTH; ++u) {
            x[u] /= diag;
        }
    }
}

//
// Light GPU Alloc Wrapper
//

struct GPUAlloc {
    GPUAlloc(::sycl::queue &q, size_t gpu_allocN) :
        q(q),
        mem(::sycl::malloc_device<real_type>(gpu_allocN, q)) {
    }

    ~GPUAlloc() {
        ::sycl::free(mem, q);
    }

    ::sycl::queue &q;
    real_type *mem = nullptr;
};

//
// Implementation
//

std::pair<soa_matrix<real_type>, std::vector<unsigned long long>> csvm::cholesky(const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, const real_type jitter) const {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(!B.empty(), "The right-hand sides must not be empty!");
    PLSSVM_ASSERT(A.size() == 1, "Should be 1 as only one device.");
    PLSSVM_ASSERT(jitter >= 0., "Jitter must be over or equal to 0.0.");

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("cholesky start");

    //
    // Setup
    //

    // Get ptr to matrix data on device
    const auto &A_d = detail::move_only_any_cast<const adaptivecpp::detail::device_ptr<real_type> &>(A[0]);
    real_type *gpuA = A_d.get();

    // Create own SYCL queue
    ::sycl::queue q(::sycl::gpu_selector_v);  // This thing... specifing inorder just does nothing lol XDDDD
                                              // Not even a warning!!!! Just a sad depressing crash

    // Constants
    const auto N = B.num_cols();             // Matrix and vector size
    const auto Nb = B.num_rows();            // Vectors amount
    const auto Ndiv = ceil_div(N, WIDTH);    // How many blocks are needed to fit whole N
    const auto Nbdiv = ceil_div(Nb, WIDTH);  // How many blocks are needed to fit whole Nb
    const auto Naligned = Ndiv * WIDTH;      // Number of entries that fit whole N AND is divisible by WIDTH
    const auto Nbaligned = Nbdiv * WIDTH;    // Number of entries that fit whole Nb AND is divisible by WIDTH

    // Index functions
    const auto matindex = [=](size_t i, size_t j) -> size_t {
        return blockcol_idx<WIDTH>(N, i, j);
    };

    const auto vecindex = [=](size_t i, size_t j) -> size_t {
        return blockvec_idx(Naligned, i, j);
    };

    // Allocate and Copy B matrix
    std::vector<real_type> paddedB(blockvec_idx_alloc<WIDTH>(Ndiv, Nbdiv));  // soa_matrix.data() does NOT store data as I was told :(((
                                                                             // It's not row for row but it packs them in structlike
                                                                             // bundles and stores them after eachother in an array.
                                                                             // Like an array of structs (AoS) ... huh
                                                                             // Am I miss understanding something?
    for (size_t i = 0; i < B.num_rows(); ++i) {
        for (size_t j = 0; j < B.num_cols(); ++j) {
            paddedB[vecindex(i, j)] = B.at(i, j);
        }
    }

    auto wrappedgpuB = GPUAlloc(q, paddedB.size());
    ::sycl::event e_Bloaded = q.memcpy(wrappedgpuB.mem, paddedB.data(), paddedB.size() * sizeof(real_type));
    real_type *gpuB = wrappedgpuB.mem;

    // Event
    ::sycl::event e;

    //
    // Cholesky Decomposition
    //

    for (size_t offset = 0; offset < Ndiv - 1; ++offset) {
        const auto blockwidth = Ndiv - offset;

        const auto idxgroup = [=](size_t i, size_t j) -> size_t {
            return matindex(i + offset * WIDTH, j + offset * WIDTH);
        };

        e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(e);
            auto cholblk = ::sycl::local_accessor<real_type, 2>(::sycl::range(WIDTH, WIDTH), cgh);
            auto prevrslt = ::sycl::local_accessor<real_type, 2>(::sycl::range(CHOL_THREADCOLWIDTH, WIDTH), cgh);

            cgh.parallel_for(
                ::sycl::nd_range{ ::sycl::range(WIDTH, ceil_div((blockwidth - 1), CHOL_THREADCOLWIDTH) * WIDTH), ::sycl::range(WIDTH, WIDTH) },
                [=](::sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupj = g.get_group_id(1);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto offsetj = groupj * CHOL_THREADCOLWIDTH + 1;

                    cholblk[locali][localj] = gpuA[idxgroup(locali, localj)];
                    ::sycl::group_barrier(idx.get_group());

                    cholesky_decomposition<WIDTH>(g, cholblk, jitter);
                    ::sycl::group_barrier(idx.get_group());

                    real_type x[CHOL_THREADCOLWIDTH];
                    for (size_t k = 0; k < CHOL_THREADCOLWIDTH; ++k) {
                        if (auto offset = offsetj + k; offset < blockwidth) {
                            x[k] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                        }
                    }

                    solve_forwardsub<WIDTH, CHOL_THREADCOLWIDTH>(g, cholblk, prevrslt, x);

                    for (size_t k = 0; k < CHOL_THREADCOLWIDTH; ++k) {
                        if (auto offset = offsetj + k; offset < blockwidth) {
                            gpuA[idxgroup(locali, localj + offset * WIDTH)] = x[k];
                        }
                    }
                });
        });

        e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(e);
            auto topblk = ::sycl::local_accessor<real_type, 3>(::sycl::range(CHOL_THREADSUBWIDTH, WIDTH, WIDTH), cgh);
            auto leftblk = ::sycl::local_accessor<real_type, 3>(::sycl::range(CHOL_THREADSUBHEIGHT, WIDTH, WIDTH), cgh);

            cgh.parallel_for(
                ::sycl::nd_range{
                    ::sycl::range(ceil_div(blockwidth - 1, CHOL_THREADSUBHEIGHT) * WIDTH, ceil_div(blockwidth - 1, CHOL_THREADSUBWIDTH) * WIDTH),
                    ::sycl::range(WIDTH, WIDTH) },
                [=](::sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto groupj = g.get_group_id(1);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    if (groupi * CHOL_THREADSUBHEIGHT >= (groupj + 1) * CHOL_THREADSUBWIDTH) {
                        return;
                    }

                    const auto offseti = groupi * CHOL_THREADSUBHEIGHT + 1;
                    const auto offsetj = groupj * CHOL_THREADSUBWIDTH + 1;

                    for (size_t k = 0; k < CHOL_THREADSUBWIDTH; ++k) {
                        if (auto offset = offsetj + k; offset < blockwidth) {
                            topblk[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                        }
                    }

                    for (size_t k = 0; k < CHOL_THREADSUBHEIGHT; ++k) {
                        if (auto offset = offseti + k; offset < blockwidth) {
                            leftblk[k][locali][localj] = gpuA[idxgroup(locali, localj + offset * WIDTH)];
                        }
                    }

                    ::sycl::group_barrier(idx.get_group());

                    real_type sum[CHOL_THREADSUBHEIGHT * CHOL_THREADSUBWIDTH] = { 0 };
                    trans_matmul<WIDTH, CHOL_THREADSUBWIDTH, CHOL_THREADSUBHEIGHT>(g, leftblk, topblk, sum);

                    for (size_t i = 0; i < CHOL_THREADSUBHEIGHT; ++i) {
                        if (auto offsetii = offseti + i; offsetii < blockwidth) {
                            for (size_t j = 0; j < CHOL_THREADSUBWIDTH; ++j) {
                                if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                    if (offsetii <= offsetjj) {
                                        gpuA[idxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[i * CHOL_THREADSUBWIDTH + j];
                                    }
                                }
                            }
                        }
                    }
                });
        });
    }

    e = q.submit([&](::sycl::handler &cgh) {
        cgh.depends_on(e);
        auto cache = ::sycl::local_accessor<real_type, 2>(::sycl::range(WIDTH, WIDTH), cgh);

        cgh.parallel_for(::sycl::nd_range{ ::sycl::range(WIDTH, Naligned), ::sycl::range(WIDTH, WIDTH) }, [=](::sycl::nd_item<2> idx) {
            const auto g = idx.get_group();

            const auto groupj = g.get_group_id(1);
            const auto locali = g.get_local_id(0);
            const auto localj = g.get_local_id(1);

            cache[locali][localj] = gpuA[matindex(locali + groupj * WIDTH, localj + groupj * WIDTH)];
            ::sycl::group_barrier(idx.get_group());

            cholesky_decomposition<WIDTH>(g, cache, jitter);
            ::sycl::group_barrier(idx.get_group());

            gpuA[matindex(locali + groupj * WIDTH, localj + groupj * WIDTH)] = cache[locali][localj];
        });
    });

    //
    // Forward Substitution
    //

    e = q.submit([&](::sycl::handler &cgh) {  // Transpose blocks
        cgh.depends_on({ e_Bloaded, e });
        auto blk = ::sycl::local_accessor<real_type, 2>(::sycl::range(WIDTH, WIDTH + 1), cgh);

        cgh.parallel_for(
            ::sycl::nd_range{
                ::sycl::range(Nbaligned, Naligned),
                ::sycl::range(WIDTH, WIDTH) },
            [=](::sycl::nd_item<2> idx) {
                transpose<WIDTH>(idx, gpuB, blk);
            });
    });

    for (size_t offset = 0; offset < Ndiv; ++offset) {
        const auto blockwidth = Ndiv - offset;

        const auto matidxgroup = [=](size_t i, size_t j) -> size_t {
            return matindex(i + offset * WIDTH, j + offset * WIDTH);
        };

        const auto vecidxgroup = [=](size_t i, size_t j) -> size_t {
            return vecindex(i, j + offset * WIDTH);
        };

        e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(e);
            auto triagblk = ::sycl::local_accessor<real_type, 2>(::sycl::range(WIDTH, WIDTH), cgh);
            auto cacheline = ::sycl::local_accessor<real_type, 2>(::sycl::range(FSUB_THREADCOLHEIGHT, WIDTH), cgh);

            cgh.parallel_for(
                ::sycl::nd_range{
                    ::sycl::range(ceil_div(Nbdiv, FSUB_THREADCOLHEIGHT) * WIDTH, WIDTH),
                    ::sycl::range(WIDTH, WIDTH) },
                [=](::sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto vecoffseti = groupi * FSUB_THREADCOLHEIGHT;

                    triagblk[locali][localj] = gpuA[matidxgroup(locali, localj)];
                    ::sycl::group_barrier(g);

                    real_type x[FSUB_THREADCOLHEIGHT];
                    for (size_t k = 0; k < FSUB_THREADCOLHEIGHT; ++k) {
                        if (auto offset = vecoffseti + k; offset < Nbdiv) {
                            x[k] = gpuB[vecidxgroup(locali + offset * WIDTH, localj)];
                        }
                    }

                    solve_forwardsub<WIDTH>(g, triagblk, cacheline, x);

                    for (size_t k = 0; k < FSUB_THREADCOLHEIGHT; ++k) {
                        if (auto offset = vecoffseti + k; offset < Nbdiv) {
                            gpuB[vecidxgroup(locali + offset * WIDTH, localj)] = x[k];
                        }
                    }
                });
        });

        if (offset >= Ndiv - 1) {
            break;
        }

        e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(e);
            auto topblk = ::sycl::local_accessor<real_type, 3>(::sycl::range(FSUB_THREAD_B_LEN, WIDTH, WIDTH), cgh);
            auto leftblk = ::sycl::local_accessor<real_type, 3>(::sycl::range(FSUB_THREAD_L_LEN, WIDTH, WIDTH), cgh);

            cgh.parallel_for(
                ::sycl::nd_range{
                    ::sycl::range(ceil_div(Nbdiv, FSUB_THREAD_B_LEN) * WIDTH, ceil_div(blockwidth - 1, FSUB_THREAD_L_LEN) * WIDTH),
                    ::sycl::range(WIDTH, WIDTH) },
                [=](::sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto groupj = g.get_group_id(1);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto offseti = groupi * FSUB_THREAD_B_LEN;
                    const auto offsetj = groupj * FSUB_THREAD_L_LEN + 1;

                    for (size_t k = 0; k < FSUB_THREAD_L_LEN; ++k) {
                        if (const auto offset = offsetj + k; offset < blockwidth) {
                            leftblk[k][locali][localj] = gpuA[matidxgroup(locali, localj + offset * WIDTH)];
                        }
                    }

                    for (size_t k = 0; k < FSUB_THREAD_B_LEN; ++k) {
                        if (const auto offset = offseti + k; offset < Nbdiv) {
                            topblk[k][locali][localj] = gpuB[vecidxgroup(locali + offset * WIDTH, localj)];
                        }
                    }

                    ::sycl::group_barrier(g);

                    real_type sum[FSUB_THREAD_B_LEN * FSUB_THREAD_L_LEN] = { 0 };
                    trans_matmul<WIDTH, FSUB_THREAD_B_LEN, FSUB_THREAD_L_LEN>(g, leftblk, topblk, sum);

                    for (size_t i = 0; i < FSUB_THREAD_B_LEN; ++i) {
                        if (auto offsetii = offseti + i; offsetii < Nbdiv) {
                            for (size_t j = 0; j < FSUB_THREAD_L_LEN; ++j) {
                                if (auto offsetjj = offsetj + j; offsetjj < blockwidth) {
                                    gpuB[vecidxgroup(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[j * FSUB_THREAD_B_LEN + i];
                                }
                            }
                        }
                    }
                });
        });
    }

    //
    // Backward Substitution
    //

    for (size_t _offset = 0; _offset < Ndiv; ++_offset) {
        const auto offset = Ndiv - 1 - _offset;

        e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(e);
            auto triagblk = ::sycl::local_accessor<real_type, 2>(::sycl::range(WIDTH, WIDTH), cgh);
            auto prevrslt = ::sycl::local_accessor<real_type, 2>(::sycl::range(BSUB_THREADCOLHEIGHT, WIDTH), cgh);

            cgh.parallel_for(
                ::sycl::nd_range{
                    ::sycl::range(ceil_div(Nbdiv, BSUB_THREADCOLHEIGHT) * WIDTH, WIDTH),
                    ::sycl::range(WIDTH, WIDTH) },
                [=](::sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto vecoffseti = groupi * BSUB_THREADCOLHEIGHT;
                    const auto vecoffsetj = offset;
                    const auto matoffseti = offset;
                    const auto matoffsetj = offset;

                    {
                        const auto v = gpuA[matindex(locali + matoffseti * WIDTH, localj + matoffsetj * WIDTH)];  // Explicit load
                        triagblk[locali][localj] = localj + matoffsetj * WIDTH < N ? v : 1;                       // to make the compiler avoid branch (possibly cmov?)
                    }

                    ::sycl::group_barrier(g);

                    real_type x[BSUB_THREADCOLHEIGHT];
                    for (size_t k = 0; k < BSUB_THREADCOLHEIGHT; ++k) {
                        if (auto offsetii = vecoffseti + k; offsetii < Nbdiv) {
                            const auto v = gpuB[vecindex(locali + offsetii * WIDTH, localj + vecoffsetj * WIDTH)];
                            x[k] = locali + vecoffsetj * WIDTH < N ? v : 0;
                        }
                    }

                    solve_backwardsub<WIDTH>(g, triagblk, prevrslt, x);

                    for (size_t k = 0; k < BSUB_THREADCOLHEIGHT; ++k) {
                        if (auto offsetii = vecoffseti + k; offsetii < Nbdiv) {
                            gpuB[vecindex(locali + offsetii * WIDTH, localj + vecoffsetj * WIDTH)] = x[k];
                        }
                    }
                });
        });

        if (_offset >= Ndiv - 1) {
            break;
        }

        e = q.submit([&](::sycl::handler &cgh) {
            cgh.depends_on(e);
            auto topblk = ::sycl::local_accessor<real_type, 3>(::sycl::range(BSUB_THREAD_B_LEN, WIDTH, WIDTH), cgh);
            auto leftblk = ::sycl::local_accessor<real_type, 3>(::sycl::range(BSUB_THREAD_L_LEN, WIDTH, WIDTH + 1), cgh);

            cgh.parallel_for(
                ::sycl::nd_range{
                    ::sycl::range(ceil_div(Nbdiv, BSUB_THREAD_B_LEN) * WIDTH, ceil_div(offset, BSUB_THREAD_L_LEN) * WIDTH),
                    ::sycl::range(WIDTH, WIDTH) },
                [=](::sycl::nd_item<2> idx) {
                    const auto g = idx.get_group();

                    const auto groupi = g.get_group_id(0);
                    const auto groupj = g.get_group_id(1);
                    const auto locali = g.get_local_id(0);
                    const auto localj = g.get_local_id(1);

                    const auto offseti = groupi * BSUB_THREAD_B_LEN;
                    const auto offsetj = groupj * BSUB_THREAD_L_LEN;

                    for (size_t k = 0; k < BSUB_THREAD_L_LEN; ++k) {
                        if (const auto offsetjj = offsetj + k; offsetjj < offset) {
                            const auto x = gpuA[matindex(locali + offsetjj * WIDTH, localj + offset * WIDTH)];
                            leftblk[k][localj][locali] = localj + offset * WIDTH < N ? x : 0;  // Transpose & filter out
                        }
                    }

                    for (size_t k = 0; k < BSUB_THREAD_B_LEN; ++k) {
                        if (const auto offsetii = offseti + k; offsetii < Nbdiv) {
                            topblk[k][locali][localj] = gpuB[vecindex(locali + offsetii * WIDTH, localj + offset * WIDTH)];  // Tail already contains zeros from before
                        }
                    }

                    ::sycl::group_barrier(g);

                    real_type sum[BSUB_THREAD_B_LEN * BSUB_THREAD_L_LEN] = { 0 };
                    trans_matmul<WIDTH, BSUB_THREAD_B_LEN, BSUB_THREAD_L_LEN>(g, leftblk, topblk, sum);

                    for (size_t i = 0; i < BSUB_THREAD_B_LEN; ++i) {
                        if (auto offsetii = offseti + i; offsetii < Nbdiv) {
                            for (size_t j = 0; j < BSUB_THREAD_L_LEN; ++j) {
                                if (auto offsetjj = offsetj + j; offsetjj < offset) {
                                    gpuB[vecindex(locali + offsetii * WIDTH, localj + offsetjj * WIDTH)] -= sum[j * BSUB_THREAD_B_LEN + i];
                                }
                            }
                        }
                    }
                });
        });
    }

    e = q.submit([&](::sycl::handler &cgh) {  // Transpose blocks back
        cgh.depends_on(e);
        auto cacheblock = ::sycl::local_accessor<real_type, 2>(::sycl::range(WIDTH, WIDTH + 1), cgh);

        cgh.parallel_for(
            ::sycl::nd_range{
                ::sycl::range(Nbaligned, Naligned),
                ::sycl::range(WIDTH, WIDTH) },
            [=](::sycl::nd_item<2> idx) {
                const auto globalij = idx.get_global_linear_id();
                const auto locali = idx.get_local_id(0);
                const auto localj = idx.get_local_id(1);

                cacheblock[locali][localj] = gpuB[globalij];

                ::sycl::group_barrier(idx.get_group());

                gpuB[globalij] = cacheblock[localj][locali];
            });
    });

    //
    // Collect Data and Return
    //

    q.memcpy(paddedB.data(), gpuB, paddedB.size() * sizeof(real_type), e).wait_and_throw();

    auto X = soa_matrix<real_type>(B.shape());  // This has a constructor taking in a padding in form of a shape with the parameters x, y
                                                // x is the width ... right? ... oh soa_matrix.num_rows() returns the x......... :(
                                                // Or maybe I'm just really blind
    for (size_t i = 0; i < X.num_rows(); ++i) {
        for (size_t j = 0; j < X.num_cols(); ++j) {
            X.at(i, j) = paddedB[vecindex(i, j)];  // Use .at(). trust nothing. everything here is out to get me
        }
    }

    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("cholesky end");

    return std::make_pair(X, std::vector<unsigned long long>(N, 0));  // Holy **** it **** works :)))))))
}

// ------------------------------------
// ------------------------------------
// ------------------------------------
// End of Cholesky Implementation
// ------------------------------------
// ------------------------------------
// ------------------------------------

std::pair<std::vector<real_type>, real_type> csvm::perform_dimensional_reduction(const parameter &params, const soa_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix must not be empty!");

    const std::chrono::steady_clock::time_point dimension_reduction_start_time = std::chrono::steady_clock::now();

    const std::size_t num_rows_reduced = A.num_rows() - 1;

    // create q_red vector and calculate QA_costs
    std::vector<real_type> q_red(num_rows_reduced);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
#pragma omp parallel for default(none) shared(q_red, A) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::linear>(A, i, A, num_rows_reduced);
            }
            break;
        case kernel_function_type::polynomial:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::polynomial>(A, i, A, num_rows_reduced, params.degree, std::get<real_type>(params.gamma), params.coef0);
            }
            break;
        case kernel_function_type::rbf:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::rbf>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma));
            }
            break;
        case kernel_function_type::sigmoid:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::sigmoid>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma), params.coef0);
            }
            break;
        case kernel_function_type::laplacian:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::laplacian>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma));
            }
            break;
        case kernel_function_type::chi_squared:
#pragma omp parallel for default(none) shared(q_red, A, params) firstprivate(num_rows_reduced)
            for (std::size_t i = 0; i < num_rows_reduced; ++i) {
                q_red[i] = kernel_function<kernel_function_type::chi_squared>(A, i, A, num_rows_reduced, std::get<real_type>(params.gamma));
            }
            break;
    }
    const real_type QA_cost = kernel_function(A, num_rows_reduced, A, num_rows_reduced, params) + real_type{ 1.0 } / params.cost;
    const std::chrono::steady_clock::time_point dimension_reduction_end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Performed dimensional reduction in {}.\n",
                detail::tracking::tracking_entry{ "cg", "dimensional_reduction", std::chrono::duration_cast<std::chrono::milliseconds>(dimension_reduction_end_time - dimension_reduction_start_time) });

    return std::make_pair(std::move(q_red), QA_cost);
}

std::chrono::duration<long, std::milli> csvm::run_blas_level_3(const solver_type cg_solver, const real_type alpha, const std::vector<detail::move_only_any> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const {
    PLSSVM_ASSERT(!B.empty(), "The B matrix must not be empty!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix must not be empty!");

    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    this->blas_level_3(cg_solver, alpha, A, B, beta, C);

    const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
}

aos_matrix<real_type> csvm::run_predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const {
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    decltype(auto) res = this->predict_values(params, support_vectors, alpha, rho, w, predict_points);

    const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Predicted the values of {} predict points using {} support vectors with {} features each in {}.\n",
                predict_points.num_rows(),
                support_vectors.num_rows(),
                support_vectors.num_cols(),
                detail::tracking::tracking_entry{ "predict_values", "total_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

    return res;
}

}  // namespace plssvm
