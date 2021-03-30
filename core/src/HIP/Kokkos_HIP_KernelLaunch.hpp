/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_HIP_KERNEL_LAUNCH_HPP
#define KOKKOS_HIP_KERNEL_LAUNCH_HPP

#include <Kokkos_Macros.hpp>

#if defined(__HIPCC__)

#include <HIP/Kokkos_HIP_Error.hpp>
#include <HIP/Kokkos_HIP_Instance.hpp>
#include <HIP/Kokkos_HIP_GraphNodeKernel.hpp>
#include <impl/Kokkos_GraphImpl_fwd.hpp>
#include <Kokkos_HIP_Space.hpp>

// Must use global variable on the device with HIP-Clang
#ifdef __HIP__
__device__ __constant__ unsigned long kokkos_impl_hip_constant_memory_buffer
    [Kokkos::Experimental::Impl::HIPTraits::ConstantMemoryUsage /
     sizeof(unsigned long)];
#endif

namespace Kokkos {
namespace Experimental {
template <typename T>
inline __device__ T *kokkos_impl_hip_shared_memory() {
  HIP_DYNAMIC_SHARED(HIPSpace::size_type, sh);
  return (T *)sh;
}
}  // namespace Experimental
}  // namespace Kokkos

namespace Kokkos {
namespace Experimental {
namespace Impl {

template <typename DriverType>
__global__ static void hip_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_hip_constant_memory_buffer));

  driver();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ __launch_bounds__(
    maxTperB, minBperSM) static void hip_parallel_launch_constant_memory() {
  const DriverType &driver = *(reinterpret_cast<const DriverType *>(
      kokkos_impl_hip_constant_memory_buffer));

  driver();
}

template <class DriverType>
__global__ static void hip_parallel_launch_local_memory(
    const DriverType *driver) {
  // FIXME_HIP driver() pass by copy
  driver->operator()();
}

template <class DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ __launch_bounds__(
    maxTperB,
    minBperSM) static void hip_parallel_launch_local_memory(const DriverType
                                                                *driver) {
  // FIXME_HIP driver() pass by copy
  driver->operator()();
}

template <typename DriverType>
__global__ static void hip_parallel_launch_global_memory(
    const DriverType *driver) {
  driver->operator()();
}

template <typename DriverType, unsigned int maxTperB, unsigned int minBperSM>
__global__ __launch_bounds__(
    maxTperB,
    minBperSM) static void hip_parallel_launch_global_memory(const DriverType
                                                                 *driver) {
  driver->operator()();
}

enum class HIPLaunchMechanism : unsigned {
  Default        = 0,
  ConstantMemory = 1,
  GlobalMemory   = 2,
  LocalMemory    = 4
};

constexpr inline HIPLaunchMechanism operator|(HIPLaunchMechanism p1,
                                              HIPLaunchMechanism p2) {
  return static_cast<HIPLaunchMechanism>(static_cast<unsigned>(p1) |
                                         static_cast<unsigned>(p2));
}
constexpr inline HIPLaunchMechanism operator&(HIPLaunchMechanism p1,
                                              HIPLaunchMechanism p2) {
  return static_cast<HIPLaunchMechanism>(static_cast<unsigned>(p1) &
                                         static_cast<unsigned>(p2));
}

template <HIPLaunchMechanism l>
struct HIPDispatchProperties {
  HIPLaunchMechanism launch_mechanism = l;
};

// Use local memory up to ConstantMemoryUseThreshold
// Use global memory above ConstantMemoryUsage
// In between use ConstantMemoryo
template <typename DriverType>
struct DeduceHIPLaunchMechanism {
  constexpr static const Kokkos::Experimental::WorkItemProperty::
      HintLightWeight_t light_weight =
          Kokkos::Experimental::WorkItemProperty::HintLightWeight;
  constexpr static const Kokkos::Experimental::WorkItemProperty::
      HintHeavyWeight_t heavy_weight =
          Kokkos::Experimental::WorkItemProperty::HintHeavyWeight;
  constexpr static const typename DriverType::Policy::work_item_property
      property = typename DriverType::Policy::work_item_property();

  static constexpr const HIPLaunchMechanism valid_launch_mechanism =
      // BuildValidMask
      (sizeof(DriverType) < HIPTraits::KernelArgumentLimit
           ? HIPLaunchMechanism::LocalMemory
           : HIPLaunchMechanism::Default) |
      (sizeof(DriverType) < HIPTraits::ConstantMemoryUsage
           ? HIPLaunchMechanism::ConstantMemory
           : HIPLaunchMechanism::Default) |
      HIPLaunchMechanism::GlobalMemory;

  static constexpr const HIPLaunchMechanism requested_launch_mechanism =
      (((property & light_weight) == light_weight)
           ? HIPLaunchMechanism::LocalMemory
           : HIPLaunchMechanism::ConstantMemory) |
      HIPLaunchMechanism::GlobalMemory;

  static constexpr const HIPLaunchMechanism default_launch_mechanism =
      // BuildValidMask
      (sizeof(DriverType) < HIPTraits::ConstantMemoryUseThreshold)
          ? HIPLaunchMechanism::LocalMemory
          : ((sizeof(DriverType) < HIPTraits::ConstantMemoryUsage)
                 ? HIPLaunchMechanism::ConstantMemory
                 : HIPLaunchMechanism::GlobalMemory);

  //              None                LightWeight    HeavyWeight
  // F<UseT       LCG  LCG L  L       LCG  LG L  L   LCG  CG L  C
  // UseT<F<KAL   LCG  LCG C  C       LCG  LG C  L   LCG  CG C  C
  // Kal<F<CMU     CG  LCG C  C        CG  LG C  G    CG  CG C  C
  // CMU<F          G  LCG G  G         G  LG G  G     G  CG G  G
  static constexpr const HIPLaunchMechanism launch_mechanism =
      ((property & light_weight) == light_weight)
          ? (sizeof(DriverType) < HIPTraits::KernelArgumentLimit
                 ? HIPLaunchMechanism::LocalMemory
                 : HIPLaunchMechanism::GlobalMemory)
          : (((property & heavy_weight) == heavy_weight)
                 ? (sizeof(DriverType) < HIPTraits::ConstantMemoryUsage
                        ? HIPLaunchMechanism::ConstantMemory
                        : HIPLaunchMechanism::GlobalMemory)
                 : (default_launch_mechanism));
};

//---------------------------------------------------------------//
// HIPParallelLaunchKernelFunc structure and its specializations //
//---------------------------------------------------------------//
template <typename DriverType, typename LaunchBounds,
          HIPLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelFunc;

// HIPLaunchMechanism::LocalMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    HIPLaunchMechanism::LocalMemory> {
  static auto get_kernel_func() {
    return hip_parallel_launch_local_memory<DriverType, MaxThreadsPerBlock,
                                            MinBlocksPerSM>;
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   HIPLaunchMechanism::LocalMemory> {
  static auto get_kernel_func() {
    return hip_parallel_launch_local_memory<DriverType, 1024, 1>;
  }
};

// HIPLaunchMechanism::GlobalMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    HIPLaunchMechanism::GlobalMemory> {
  static auto get_kernel_func() {
    return hip_parallel_launch_global_memory<DriverType, MaxThreadsPerBlock,
                                             MinBlocksPerSM>;
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   HIPLaunchMechanism::GlobalMemory> {
  static auto get_kernel_func() {
    return hip_parallel_launch_global_memory<DriverType>;
  }
};

// HIPLaunchMechanism::ConstantMemory specializations
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM>
struct HIPParallelLaunchKernelFunc<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    HIPLaunchMechanism::ConstantMemory> {
  static auto get_kernel_func() {
    return hip_parallel_launch_constant_memory<DriverType, MaxThreadsPerBlock,
                                               MinBlocksPerSM>;
  }
};

template <typename DriverType>
struct HIPParallelLaunchKernelFunc<DriverType, Kokkos::LaunchBounds<0, 0>,
                                   HIPLaunchMechanism::ConstantMemory> {
  static auto get_kernel_func() {
    return hip_parallel_launch_constant_memory<DriverType>;
  }
};

//------------------------------------------------------------------//
// HIPParallelLaunchKernelInvoker structure and its specializations //
//------------------------------------------------------------------//
template <typename DriverType, typename LaunchBounds,
          HIPLaunchMechanism LaunchMechanism>
struct HIPParallelLaunchKernelInvoker;

// HIPLaunchMechanism::LocalMemory specialization
template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      HIPLaunchMechanism::LocalMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  HIPLaunchMechanism::LocalMemory> {
  using base_t = HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                             HIPLaunchMechanism::LocalMemory>;

  static void invoke_kernel(DriverType const *driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            HIPInternal const *hip_instance) {
    (base_t::get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>(
        driver);
  }

#ifdef KOKKOS_HIP_ENABLE_GRAPHS
  // FIXME
  inline static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block,
      int /*shmem*/, HIPInternal const * /*hip_instance*/,
      bool /*prefer_shmem*/) {
    auto const &graph = ::Kokkos::Impl::get_hip_graph_from_kernel(driver);
    KOKKOS_EXPECTS(bool(graph));
    auto &graph_node = ::Kokkos::Impl::get_hip_graph_node_from_kernel(driver);
    // Expect node not yet initialized
    KOKKOS_EXPECTS(!bool(graph_node));

    // TODO I need to get the gpu_executor from somewhere
    dagee::GpuExecutorAtmi gpu_executor;
    auto registered_kernel =
        gpu_executor.registerKernel<DriverType>(base_t::get_kernel_func());
    graph_node->node_details::task = std::make_unique<
        dagee::ATMIgpuKernelInstance<dagee::StdAllocatorFactory<>>>(
        gpu_executor.makeTask(grid, block, registered_kernel, driver));
  }
#endif
};

// HIPLaunchMechanism::GlobalMemory specialization
template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      HIPLaunchMechanism::GlobalMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  HIPLaunchMechanism::GlobalMemory> {
  using base_t = HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                             HIPLaunchMechanism::GlobalMemory>;

  static void invoke_kernel(DriverType const *driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            HIPInternal const *hip_instance) {
    (base_t::get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>(
        driver);
  }

  // FIXME
#ifdef KOKKOS_HIP_ENABLE_GRAPHS
  inline static void create_parallel_launch_graph_node(
      DriverType const &driver, dim3 const &grid, dim3 const &block,
      int /*shmem*/, HIPInternal const * /*hip_instance*/,
      bool /*prefer_shmem*/) {
    // TODO temporary hack
    auto const &graph = ::Kokkos::Impl::get_hip_graph_from_kernel(driver);
    KOKKOS_EXPECTS(bool(graph));
    auto &graph_node = ::Kokkos::Impl::get_hip_graph_node_from_kernel(driver);
    // Expect node not yet initialized
    KOKKOS_EXPECTS(!bool(graph_node));

    // TODO I need to get the gpu_executor from somewhere
    dagee::GpuExecutorAtmi gpu_executor;
    auto registered_kernel =
        gpu_executor.registerKernel<DriverType>(base_t::get_kernel_func());
    graph_node->node_details::task =
        gpu_executor.makeTask(grid, block, registered_kernel, driver);
  }
#endif
};

// HIPLaunchMechanism::ConstantMemory specializations
template <typename DriverType, typename LaunchBounds>
struct HIPParallelLaunchKernelInvoker<DriverType, LaunchBounds,
                                      HIPLaunchMechanism::ConstantMemory>
    : HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  HIPLaunchMechanism::ConstantMemory> {
  using base_t =
      HIPParallelLaunchKernelFunc<DriverType, LaunchBounds,
                                  HIPLaunchMechanism::ConstantMemory>;
  static_assert(sizeof(DriverType) < HIPTraits::ConstantMemoryUsage,
                "Kokkos Error: Requested HIPLaunchConstantMemory with a "
                "Functor larger than 32kB.");

  static void invoke_kernel(DriverType const *driver, dim3 const &grid,
                            dim3 const &block, int shmem,
                            HIPInternal const *hip_instance) {
    // Wait until the previous kernel that uses the constant buffer is done
    HIP_SAFE_CALL(hipEventSynchronize(hip_instance->constantMemReusable));

    // Copy functor (synchronously) to staging buffer in pinned host memory
    unsigned long *staging = hip_instance->constantMemHostStaging;
    memcpy(staging, driver, sizeof(DriverType));

    // Copy functor asynchronously from there to constant memory on the device
    HIP_SAFE_CALL(hipMemcpyToSymbolAsync(
        HIP_SYMBOL(kokkos_impl_hip_constant_memory_buffer), staging,
        sizeof(DriverType), 0, hipMemcpyHostToDevice,
        hipStream_t(hip_instance->m_stream)));

    // Invoke the driver function on the device
    (base_t::
         get_kernel_func())<<<grid, block, shmem, hip_instance->m_stream>>>();

    // Record an event that says when the constant buffer can be reused
    HIP_SAFE_CALL(hipEventRecord(hip_instance->constantMemReusable,
                                 hipStream_t(hip_instance->m_stream)));
  }
};

//-----------------------------//
// HIPParallelLaunch structure //
//-----------------------------//
template <typename DriverType, typename LaunchBounds = Kokkos::LaunchBounds<>,
          HIPLaunchMechanism LaunchMechanism =
              DeduceHIPLaunchMechanism<DriverType>::launch_mechanism,
          bool DoGraph = DriverType::Policy::is_graph_kernel::value
#ifndef KOKKOS_CUDA_ENABLE_GRAPHS
                         && false
#endif
          >
struct HIPParallelLaunch;

template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM, HIPLaunchMechanism LaunchMechanism>
struct HIPParallelLaunch<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    LaunchMechanism,
    /* DoGraph = */ false>
    : HIPParallelLaunchKernelInvoker<
          DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
          LaunchMechanism> {
  using base_t = HIPParallelLaunchKernelInvoker<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      LaunchMechanism>;

  HIPParallelLaunch(const DriverType &driver, const dim3 &grid,
                    const dim3 &block, const int shmem,
                    const HIPInternal *hip_instance,
                    const bool /*prefer_shmem*/) {
    if ((grid.x != 0) && ((block.x * block.y * block.z) != 0)) {
      if (hip_instance->m_maxShmemPerBlock < shmem) {
        Kokkos::Impl::throw_runtime_exception(
            "HIPParallelLaunch FAILED: shared memory request is too large");
      }

      KOKKOS_ENSURE_HIP_LOCK_ARRAYS_ON_DEVICE();

      // Invoke the driver function on the device
      DriverType *d_driver = reinterpret_cast<DriverType *>(
          hip_instance->get_next_driver(sizeof(DriverType)));
      std::memcpy((void *)d_driver, (void *)&driver, sizeof(DriverType));
      base_t::invoke_kernel(d_driver, grid, block, shmem, hip_instance);

#if defined(KOKKOS_ENABLE_DEBUG_BOUNDS_CHECK)
      HIP_SAFE_CALL(hipGetLastError());
      hip_instance->fence();
#endif
    }
  }

  static hipFuncAttributes get_hip_func_attributes() {
    static hipFuncAttributes attr = []() {
      hipFuncAttributes attr;
      HIP_SAFE_CALL(hipFuncGetAttributes(
          &attr, reinterpret_cast<void const *>(base_t::get_kernel_func())));
      return attr;
    }();
    return attr;
  }
};

#ifdef KOKKOS_HIP_ENABLE_GRAPHS
template <typename DriverType, unsigned int MaxThreadsPerBlock,
          unsigned int MinBlocksPerSM, HIPLaunchMechanism LaunchMechanism>
struct HIPParallelLaunch<
    DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
    LaunchMechanism,
    /* DoGraph = */ true>
    : HIPParallelLaunchKernelInvoker<
          DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
          LaunchMechanism> {
  using base_t = HIPParallelLaunchKernelInvoker<
      DriverType, Kokkos::LaunchBounds<MaxThreadsPerBlock, MinBlocksPerSM>,
      LaunchMechanism>;
  template <class... Args>
  HIPParallelLaunch(Args &&...args) {
    base_t::create_parallel_launch_graph_node((Args &&) args...);
  }
};
#endif
}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#endif

#endif
