// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif

namespace TestConcept {

using ExecutionSpace = TEST_EXECSPACE;
using MemorySpace    = typename ExecutionSpace::memory_space;
using DeviceType     = typename ExecutionSpace::device_type;

// No access relationship is registered for this space, including on host-only
// builds, so it exercises the inaccessible SpaceAccessibility::space branch.
template <class ExecSpace>
struct UnrelatedMemorySpace {
  using memory_space    = UnrelatedMemorySpace;
  using execution_space = ExecSpace;
};

template <class AccessSpace, class SrcMemorySpace>
constexpr bool check_space_accessibility() {
  using ExecSpace      = typename AccessSpace::execution_space;
  using DstMemorySpace = typename AccessSpace::memory_space;
  using Access = Kokkos::SpaceAccessibility<AccessSpace, SrcMemorySpace>;
  constexpr bool accessible =
      Kokkos::Impl::MemorySpaceAccess<typename ExecSpace::memory_space,
                                      SrcMemorySpace>::accessible;
  constexpr bool assignable =
      Kokkos::is_memory_space_v<AccessSpace> &&
      Kokkos::Impl::MemorySpaceAccess<DstMemorySpace,
                                      SrcMemorySpace>::assignable;
  using ExpectedSpace = std::conditional_t<
      std::is_same_v<DstMemorySpace, SrcMemorySpace> || !accessible,
      AccessSpace, Kokkos::Device<ExecSpace, SrcMemorySpace>>;

  static_assert(Access::accessible == accessible);
  static_assert(Access::assignable == assignable);
  static_assert(std::is_same_v<typename Access::space, ExpectedSpace>);
  return true;
}

template <class DstMemorySpace, class SrcMemorySpace>
constexpr bool check_memory_space_traits() {
  using ExecSpace = typename DstMemorySpace::execution_space;
  constexpr bool assignable =
      Kokkos::Impl::MemorySpaceAccess<DstMemorySpace,
                                      SrcMemorySpace>::assignable;
  constexpr bool accessible =
      Kokkos::Impl::MemorySpaceAccess<typename ExecSpace::memory_space,
                                      SrcMemorySpace>::accessible;

  static_assert(
      Kokkos::is_assignable_mem_space<DstMemorySpace, SrcMemorySpace>{} ==
      assignable);
  static_assert(
      Kokkos::is_assignable_mem_space_v<DstMemorySpace, SrcMemorySpace> ==
      assignable);
  static_assert(Kokkos::is_assignable_mem_space_v<const DstMemorySpace,
                                                  volatile SrcMemorySpace> ==
                assignable);
  static_assert(std::is_same_v<typename Kokkos::is_assignable_mem_space<
                                   DstMemorySpace, SrcMemorySpace>::type,
                               std::bool_constant<assignable>>);
  static_assert(Kokkos::is_accessible_mem_space<ExecSpace, SrcMemorySpace>{} ==
                accessible);
  static_assert(Kokkos::is_accessible_mem_space_v<ExecSpace, SrcMemorySpace> ==
                accessible);
  static_assert(Kokkos::is_accessible_mem_space_v<const ExecSpace,
                                                  volatile SrcMemorySpace> ==
                accessible);
  static_assert(std::is_same_v<typename Kokkos::is_accessible_mem_space<
                                   ExecSpace, SrcMemorySpace>::type,
                               std::bool_constant<accessible>>);

  static_assert(check_space_accessibility<ExecSpace, SrcMemorySpace>());
  // Only instantiate SpaceAccessibility for valid execution/memory pairs.
  if constexpr (Kokkos::Impl::MemorySpaceAccess<
                    typename ExecSpace::memory_space,
                    DstMemorySpace>::accessible) {
    static_assert(check_space_accessibility<DstMemorySpace, SrcMemorySpace>());
    static_assert(
        check_space_accessibility<Kokkos::Device<ExecSpace, DstMemorySpace>,
                                  SrcMemorySpace>());
  }
  return true;
}

template <class DstMemorySpace, class... SrcMemorySpaces>
constexpr bool check_memory_space_traits_row() {
  return (check_memory_space_traits<DstMemorySpace, SrcMemorySpaces>() && ...);
}

template <class... MemorySpaces>
constexpr bool check_memory_space_traits_matrix() {
  return (check_memory_space_traits_row<MemorySpaces, MemorySpaces...>() &&
          ...);
}

static_assert(
    check_memory_space_traits_matrix<
        MemorySpace, Kokkos::HostSpace, Kokkos::AnonymousSpace,
        UnrelatedMemorySpace<ExecutionSpace>,
        typename ExecutionSpace::scratch_memory_space,
        typename Kokkos::DefaultHostExecutionSpace::scratch_memory_space
#ifdef KOKKOS_ENABLE_SERIAL
        ,
        Kokkos::Serial::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_OPENMP
        ,
        Kokkos::OpenMP::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_THREADS
        ,
        Kokkos::Threads::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_HPX
        ,
        Kokkos::Experimental::HPX::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_CUDA
        ,
        Kokkos::CudaSpace, Kokkos::CudaUVMSpace, Kokkos::CudaHostPinnedSpace,
        Kokkos::Cuda::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_HIP
        ,
        Kokkos::HIPSpace, Kokkos::HIPManagedSpace, Kokkos::HIPHostPinnedSpace,
        Kokkos::HIP::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_SYCL
        ,
        Kokkos::SYCLDeviceUSMSpace, Kokkos::SYCLSharedUSMSpace,
        Kokkos::SYCLHostUSMSpace, Kokkos::SYCL::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_OPENACC
        ,
        Kokkos::Experimental::OpenACCSpace,
        Kokkos::Experimental::OpenACC::scratch_memory_space
#endif
#ifdef KOKKOS_ENABLE_NEXTSILICON
        ,
        Kokkos::Experimental::NextSiliconSharedSpace,
        Kokkos::Experimental::NextSilicon::scratch_memory_space
#endif
        >());

template <class HostExecutionSpace>
constexpr bool check_host_memory_space_traits() {
  using ScratchSpace = typename HostExecutionSpace::scratch_memory_space;
  static_assert(
      Kokkos::is_accessible_mem_space_v<HostExecutionSpace, Kokkos::HostSpace>);
  static_assert(
      Kokkos::is_accessible_mem_space_v<HostExecutionSpace, ScratchSpace>);
  static_assert(Kokkos::is_assignable_mem_space_v<ScratchSpace, ScratchSpace>);
  static_assert(
      !Kokkos::is_assignable_mem_space_v<Kokkos::HostSpace, ScratchSpace>);
  static_assert(
      !Kokkos::is_assignable_mem_space_v<ScratchSpace, Kokkos::HostSpace>);
  return true;
}

#ifdef KOKKOS_ENABLE_SERIAL
static_assert(check_host_memory_space_traits<Kokkos::Serial>());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
static_assert(check_host_memory_space_traits<Kokkos::OpenMP>());
#endif
#ifdef KOKKOS_ENABLE_THREADS
static_assert(check_host_memory_space_traits<Kokkos::Threads>());
#endif
#ifdef KOKKOS_ENABLE_HPX
static_assert(check_host_memory_space_traits<Kokkos::Experimental::HPX>());
#endif

static_assert(
    Kokkos::is_assignable_mem_space_v<Kokkos::HostSpace, Kokkos::HostSpace>);
static_assert(
    Kokkos::is_assignable_mem_space_v<MemorySpace, Kokkos::AnonymousSpace>);
static_assert(
    Kokkos::is_assignable_mem_space_v<Kokkos::AnonymousSpace, MemorySpace>);

// AnonymousSpace is permissive only when the other argument is a valid space.
static_assert(!Kokkos::is_assignable_mem_space_v<Kokkos::AnonymousSpace, void>);
static_assert(!Kokkos::is_assignable_mem_space_v<int, Kokkos::AnonymousSpace>);
static_assert(
    !Kokkos::is_assignable_mem_space_v<Kokkos::AnonymousSpace, MemorySpace &>);
static_assert(
    !Kokkos::is_assignable_mem_space_v<MemorySpace &, Kokkos::AnonymousSpace>);
static_assert(!Kokkos::is_accessible_mem_space_v<void, Kokkos::AnonymousSpace>);
static_assert(
    !Kokkos::is_accessible_mem_space_v<MemorySpace, Kokkos::AnonymousSpace>);
static_assert(!Kokkos::is_accessible_mem_space_v<ExecutionSpace &,
                                                 Kokkos::AnonymousSpace>);

#ifdef KOKKOS_ENABLE_OPENACC
template <class ExecSpace>
constexpr bool check_openacc_memory_access() {
  using OpenACC      = Kokkos::Experimental::OpenACC;
  using OpenACCSpace = Kokkos::Experimental::OpenACCSpace;
#ifdef KOKKOS_ENABLE_OPENACC_FORCE_HOST_AS_DEVICE
  constexpr bool host_as_device = true;
#else
  constexpr bool host_as_device = false;
#endif
  constexpr bool is_openacc = std::is_same_v<ExecSpace, OpenACC>;
  constexpr bool is_host =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>;
  static_assert(Kokkos::is_accessible_mem_space_v<ExecSpace, OpenACCSpace> ==
                (is_openacc || (host_as_device && is_host)));
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecSpace, Kokkos::HostSpace> ==
      (is_host || (host_as_device && is_openacc)));
  static_assert(!Kokkos::is_accessible_mem_space_v<
                ExecSpace, typename OpenACC::scratch_memory_space>);
  static_assert(
      Kokkos::is_assignable_mem_space_v<Kokkos::HostSpace, OpenACCSpace> ==
      host_as_device);
  static_assert(
      Kokkos::is_assignable_mem_space_v<OpenACCSpace, Kokkos::HostSpace> ==
      host_as_device);
  return true;
}

static_assert(check_openacc_memory_access<Kokkos::Experimental::OpenACC>());
#ifdef KOKKOS_ENABLE_SERIAL
static_assert(check_openacc_memory_access<Kokkos::Serial>());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
static_assert(check_openacc_memory_access<Kokkos::OpenMP>());
#endif
#ifdef KOKKOS_ENABLE_THREADS
static_assert(check_openacc_memory_access<Kokkos::Threads>());
#endif
#ifdef KOKKOS_ENABLE_HPX
static_assert(check_openacc_memory_access<Kokkos::Experimental::HPX>());
#endif

static_assert(
    Kokkos::is_assignable_mem_space_v<Kokkos::Experimental::OpenACCSpace,
                                      Kokkos::Experimental::OpenACCSpace>);
static_assert(!Kokkos::is_assignable_mem_space_v<
              Kokkos::Experimental::OpenACCSpace,
              Kokkos::Experimental::OpenACC::scratch_memory_space>);
static_assert(!Kokkos::is_accessible_mem_space_v<
              Kokkos::Experimental::OpenACC,
              Kokkos::DefaultHostExecutionSpace::scratch_memory_space>);
#endif

#ifdef KOKKOS_ENABLE_NEXTSILICON
template <class ExecSpace>
constexpr bool check_nextsilicon_memory_access() {
  using NextSilicon             = Kokkos::Experimental::NextSilicon;
  using SharedSpace             = Kokkos::Experimental::NextSiliconSharedSpace;
  constexpr bool is_nextsilicon = std::is_same_v<ExecSpace, NextSilicon>;
  constexpr bool is_host =
      std::is_same_v<typename ExecSpace::memory_space, Kokkos::HostSpace>;
  static_assert(Kokkos::is_accessible_mem_space_v<ExecSpace, SharedSpace> ==
                (is_nextsilicon || is_host));
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecSpace, Kokkos::HostSpace> ==
      is_host);
  static_assert(Kokkos::is_accessible_mem_space_v<
                    ExecSpace, typename NextSilicon::scratch_memory_space> ==
                is_nextsilicon);
  return true;
}

static_assert(
    check_nextsilicon_memory_access<Kokkos::Experimental::NextSilicon>());
#ifdef KOKKOS_ENABLE_SERIAL
static_assert(check_nextsilicon_memory_access<Kokkos::Serial>());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
static_assert(check_nextsilicon_memory_access<Kokkos::OpenMP>());
#endif
#ifdef KOKKOS_ENABLE_THREADS
static_assert(check_nextsilicon_memory_access<Kokkos::Threads>());
#endif
#ifdef KOKKOS_ENABLE_HPX
static_assert(check_nextsilicon_memory_access<Kokkos::Experimental::HPX>());
#endif

static_assert(!Kokkos::is_assignable_mem_space_v<
              Kokkos::HostSpace, Kokkos::Experimental::NextSiliconSharedSpace>);
static_assert(!Kokkos::is_assignable_mem_space_v<
              Kokkos::Experimental::NextSiliconSharedSpace, Kokkos::HostSpace>);
static_assert(Kokkos::is_assignable_mem_space_v<
              Kokkos::Experimental::NextSiliconSharedSpace,
              Kokkos::Experimental::NextSiliconSharedSpace>);
static_assert(!Kokkos::is_assignable_mem_space_v<
              Kokkos::Experimental::NextSiliconSharedSpace,
              Kokkos::Experimental::NextSilicon::scratch_memory_space>);
static_assert(std::is_same_v<
              Kokkos::Impl::HostMirror<
                  Kokkos::Experimental::NextSiliconSharedSpace>::device_type,
              Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                             Kokkos::Experimental::NextSiliconSharedSpace>>);
#endif

static_assert(Kokkos::is_assignable_mem_space_v<MemorySpace, MemorySpace>);
static_assert(Kokkos::is_accessible_mem_space_v<ExecutionSpace, MemorySpace>);
static_assert(
    Kokkos::is_accessible_mem_space_v<ExecutionSpace, Kokkos::AnonymousSpace>);

static_assert(!Kokkos::is_assignable_mem_space<int, MemorySpace>{});
static_assert(!Kokkos::is_assignable_mem_space_v<MemorySpace, void>);
static_assert(!Kokkos::is_assignable_mem_space_v<void, void>);
static_assert(!Kokkos::is_assignable_mem_space_v<ExecutionSpace, MemorySpace>);
static_assert(!Kokkos::is_assignable_mem_space_v<MemorySpace, DeviceType>);
static_assert(!Kokkos::is_assignable_mem_space_v<MemorySpace &, MemorySpace>);
static_assert(
    !Kokkos::is_assignable_mem_space_v<MemorySpace, const MemorySpace &>);

static_assert(!Kokkos::is_accessible_mem_space<int, MemorySpace>{});
static_assert(!Kokkos::is_accessible_mem_space_v<ExecutionSpace, void>);
static_assert(!Kokkos::is_accessible_mem_space_v<void, void>);
static_assert(!Kokkos::is_accessible_mem_space_v<MemorySpace, MemorySpace>);
static_assert(!Kokkos::is_accessible_mem_space_v<DeviceType, MemorySpace>);
static_assert(
    !Kokkos::is_accessible_mem_space_v<ExecutionSpace, ExecutionSpace>);
static_assert(
    !Kokkos::is_accessible_mem_space_v<ExecutionSpace &, MemorySpace>);
static_assert(
    !Kokkos::is_accessible_mem_space_v<ExecutionSpace, const MemorySpace &>);

static_assert(Kokkos::is_execution_space<ExecutionSpace>{});
static_assert(Kokkos::is_execution_space<ExecutionSpace const>{});
static_assert(!Kokkos::is_execution_space<ExecutionSpace &>{});
static_assert(!Kokkos::is_execution_space<ExecutionSpace const &>{});
static_assert(Kokkos::ExecutionSpace<ExecutionSpace>);
static_assert(Kokkos::ExecutionSpace<ExecutionSpace const>);
static_assert(!Kokkos::ExecutionSpace<ExecutionSpace &>);
static_assert(!Kokkos::ExecutionSpace<ExecutionSpace const &>);

static_assert(Kokkos::is_memory_space<MemorySpace>{});
static_assert(Kokkos::is_memory_space<MemorySpace const>{});
static_assert(!Kokkos::is_memory_space<MemorySpace &>{});
static_assert(!Kokkos::is_memory_space<MemorySpace const &>{});
static_assert(Kokkos::MemorySpace<MemorySpace>);
static_assert(Kokkos::MemorySpace<MemorySpace const>);
static_assert(!Kokkos::MemorySpace<MemorySpace &>);
static_assert(!Kokkos::MemorySpace<MemorySpace const &>);

static_assert(Kokkos::is_device<DeviceType>{});
static_assert(Kokkos::is_device<DeviceType const>{});
static_assert(!Kokkos::is_device<DeviceType &>{});
static_assert(!Kokkos::is_device<DeviceType const &>{});

static_assert(!Kokkos::is_device<ExecutionSpace>{});
static_assert(!Kokkos::is_device<MemorySpace>{});

static_assert(Kokkos::is_space<ExecutionSpace>{});
static_assert(Kokkos::is_space<MemorySpace>{});
static_assert(Kokkos::is_space<DeviceType>{});
static_assert(Kokkos::is_space<ExecutionSpace const>{});
static_assert(Kokkos::is_space<MemorySpace const>{});
static_assert(Kokkos::is_space<DeviceType const>{});
static_assert(!Kokkos::is_space<ExecutionSpace &>{});
static_assert(!Kokkos::is_space<MemorySpace &>{});
static_assert(!Kokkos::is_space<DeviceType &>{});

static_assert(Kokkos::is_execution_space_v<ExecutionSpace>);
static_assert(!Kokkos::is_execution_space_v<ExecutionSpace &>);
static_assert(Kokkos::ExecutionSpace<ExecutionSpace>);
static_assert(!Kokkos::ExecutionSpace<ExecutionSpace &>);

static_assert(Kokkos::is_array_layout_v<Kokkos::LayoutLeft>);
static_assert(Kokkos::is_array_layout_v<Kokkos::LayoutRight>);
static_assert(Kokkos::is_array_layout_v<Kokkos::LayoutStride>);

static_assert(
    Kokkos::is_execution_policy_v<Kokkos::RangePolicy<ExecutionSpace>>);
static_assert(Kokkos::is_execution_policy_v<
              Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>>);
static_assert(
    Kokkos::is_execution_policy_v<Kokkos::TeamPolicy<ExecutionSpace>>);
static_assert(Kokkos::ExecutionPolicy<Kokkos::RangePolicy<ExecutionSpace>>);
static_assert(Kokkos::ExecutionPolicy<
              Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>>);
static_assert(Kokkos::ExecutionPolicy<Kokkos::TeamPolicy<ExecutionSpace>>);

static_assert(Kokkos::is_reducer_v<Kokkos::Sum<int, Kokkos::HostSpace>>);
static_assert(!Kokkos::is_reducer_v<int>);
static_assert(Kokkos::Reducer<Kokkos::Sum<int, Kokkos::HostSpace>>);
static_assert(!Kokkos::Reducer<int>);

/*-------------------------------------------------
  begin test for team_handle concept

  Here we also provide a complete trait that follows the full concept specified
  in:
    https://github.com/kokkos/kokkos/blob/61d7db55fceac3318c987a291f77b844fd94c165/core/src/Kokkos_Concepts.hpp#L160
  but this is not used as implementation for performance reasons as discussed
  in: https://github.com/kokkos/kokkos/pull/5375
  ------------------------------------------------- */

template <typename T>
struct is_team_handle_complete_trait_check {
 private:
  struct TrivialFunctor {
    void operator()(double &) {}
  };
  using test_value_type = double;
  test_value_type lvalueForMethodsNeedingIt_;
  test_value_type *ptrForMethodsNeedingIt_;
  // we use Sum here but any other reducer can be used
  // since we just want something that meets the ReducerConcept
  using reduction_to_test_t = ::Kokkos::Sum<test_value_type>;

  // nested aliases
  template <class U>
  using ExecutionSpaceArchetypeAlias = typename U::execution_space;
  template <class U>
  using ScratchMemorySpaceArchetypeAlias = typename U::scratch_memory_space;

  // "indices" methods
  template <class U>
  using TeamRankArchetypeExpr = decltype(std::declval<U const &>().team_rank());

  template <class U>
  using TeamSizeArchetypeExpr = decltype(std::declval<U const &>().team_size());

  template <class U>
  using LeagueRankArchetypeExpr =
      decltype(std::declval<U const &>().league_rank());

  template <class U>
  using LeagueSizeArchetypeExpr =
      decltype(std::declval<U const &>().league_size());

  // scratch space
  template <class U>
  using TeamShmemArchetypeExpr =
      decltype(std::declval<U const &>().team_shmem());

  template <class U>
  using TeamScratchArchetypeExpr =
      decltype(std::declval<U const &>().team_scratch(0));

  template <class U>
  using ThreadScracthArchetypeExpr =
      decltype(std::declval<U const &>().thread_scratch(0));

  // collectives
  template <class U>
  using TeamBarrierArchetypeExpr =
      decltype(std::declval<U const &>().team_barrier());

  template <class U>
  using TeamBroadcastArchetypeExpr =
      decltype(std::declval<U const &>().team_broadcast(
          lvalueForMethodsNeedingIt_, 0));

  template <class U>
  using TeamBroadcastAcceptClosureArchetypeExpr =
      decltype(std::declval<U const &>().team_broadcast(
          TrivialFunctor{}, lvalueForMethodsNeedingIt_, 0));

  template <class U>
  using TeamReducedArchetypeExpr =
      decltype(std::declval<U const &>().team_reduce(
          std::declval<reduction_to_test_t>()));

  template <class U>
  using TeamScanArchetypeExpr = decltype(std::declval<U const &>().team_scan(
      lvalueForMethodsNeedingIt_, ptrForMethodsNeedingIt_));

 public:
  static constexpr bool value =
      Kokkos::is_detected_v<ExecutionSpaceArchetypeAlias, T> &&
      Kokkos::is_detected_v<ScratchMemorySpaceArchetypeAlias, T> &&
      //
      Kokkos::is_detected_exact_v<int, TeamRankArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<int, TeamSizeArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<int, LeagueRankArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<int, LeagueSizeArchetypeExpr, T> &&
      //
      Kokkos::is_detected_exact_v<
          Kokkos::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          TeamShmemArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<
          Kokkos::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          TeamScratchArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<
          Kokkos::detected_t<ScratchMemorySpaceArchetypeAlias, T> const &,
          ThreadScracthArchetypeExpr, T> &&
      //
      Kokkos::is_detected_exact_v<void, TeamBarrierArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<void, TeamBroadcastArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<void, TeamBroadcastAcceptClosureArchetypeExpr,
                                  T> &&
      Kokkos::is_detected_exact_v<void, TeamReducedArchetypeExpr, T> &&
      Kokkos::is_detected_exact_v<test_value_type, TeamScanArchetypeExpr, T>;
  constexpr operator bool() const noexcept { return value; }
};

template <class T>
inline constexpr bool is_team_handle_complete_trait_check_v =
    is_team_handle_complete_trait_check<T>::value;

// actual test begins here

/*
  FIXME_OPENACC
  OpenACCTeamMember is missing the following method:
    template <typename ReducerType>
    KOKKOS_INLINE_FUNCTION std::enable_if_t<is_reducer<ReducerType>::value>
    team_reduce(ReducerType const& reducer) const noexcept;
*/

#if !defined(KOKKOS_ENABLE_OPENACC)
using space_t  = TEST_EXECSPACE;
using policy_t = Kokkos::TeamPolicy<space_t>;
using member_t = typename policy_t::member_type;

// is_team_handle uses the actual core implementation
static_assert(Kokkos::is_team_handle<member_t>::value);
static_assert(Kokkos::is_team_handle_v<member_t>);
static_assert(Kokkos::is_team_handle_v<member_t const>);
static_assert(!Kokkos::is_team_handle_v<member_t &>);
static_assert(!Kokkos::is_team_handle_v<member_t const &>);
static_assert(!Kokkos::is_team_handle_v<member_t *>);
static_assert(!Kokkos::is_team_handle_v<member_t const *>);
static_assert(!Kokkos::is_team_handle_v<member_t *const>);
static_assert(Kokkos::TeamHandle<member_t>);
static_assert(Kokkos::TeamHandle<member_t>);
static_assert(Kokkos::TeamHandle<member_t const>);
static_assert(!Kokkos::TeamHandle<member_t &>);
static_assert(!Kokkos::TeamHandle<member_t const &>);
static_assert(!Kokkos::TeamHandle<member_t *>);
static_assert(!Kokkos::TeamHandle<member_t const *>);
static_assert(!Kokkos::TeamHandle<member_t *const>);

// is_team_handle_complete_trait_check uses the FULL trait class above
static_assert(is_team_handle_complete_trait_check<member_t>::value);
static_assert(is_team_handle_complete_trait_check_v<member_t>);
static_assert(is_team_handle_complete_trait_check_v<member_t const>);
static_assert(!is_team_handle_complete_trait_check_v<member_t &>);
static_assert(!is_team_handle_complete_trait_check_v<member_t const &>);
static_assert(!is_team_handle_complete_trait_check_v<member_t *>);
static_assert(!is_team_handle_complete_trait_check_v<member_t const *>);
static_assert(!is_team_handle_complete_trait_check_v<member_t *const>);
#endif

}  // namespace TestConcept
