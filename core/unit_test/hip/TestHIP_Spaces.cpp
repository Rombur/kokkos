// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestHIP_Category.hpp>

namespace Test {

template <class ExecutionSpace>
constexpr bool check_hip_memory_access() {
  constexpr bool is_hip = std::is_same_v<ExecutionSpace, Kokkos::HIP>;
  constexpr bool is_host =
      std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>;

#ifdef KOKKOS_IMPL_HIP_UNIFIED_MEMORY
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecutionSpace, Kokkos::HIPSpace> ==
      (is_hip || is_host));
#else
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecutionSpace, Kokkos::HIPSpace> ==
      is_hip);
#endif
  static_assert(Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                                  Kokkos::HIPManagedSpace> ==
                (is_hip || is_host));
  static_assert(Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                                  Kokkos::HIPHostPinnedSpace> ==
                (is_hip || is_host));
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecutionSpace, Kokkos::HostSpace> ==
      is_host);
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                        Kokkos::HIP::scratch_memory_space> ==
      is_hip);
  return true;
}

TEST(hip, memory_space_traits) {
  using Host    = Kokkos::HostSpace;
  using Device  = Kokkos::HIPSpace;
  using Managed = Kokkos::HIPManagedSpace;
  using Pinned  = Kokkos::HIPHostPinnedSpace;

  static_assert(Kokkos::is_assignable_mem_space_v<Host, Host>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Host, Device>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Host, Managed>);
  static_assert(Kokkos::is_assignable_mem_space_v<Host, Pinned>);

  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Host>);
  static_assert(Kokkos::is_assignable_mem_space_v<Device, Device>);
  static_assert(Kokkos::is_assignable_mem_space_v<Device, Managed>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Pinned>);

  static_assert(!Kokkos::is_assignable_mem_space_v<Managed, Host>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Managed, Device>);
  static_assert(Kokkos::is_assignable_mem_space_v<Managed, Managed>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Managed, Pinned>);

  static_assert(!Kokkos::is_assignable_mem_space_v<Pinned, Host>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Pinned, Device>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Pinned, Managed>);
  static_assert(Kokkos::is_assignable_mem_space_v<Pinned, Pinned>);

  using Scratch = Kokkos::HIP::scratch_memory_space;
  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Scratch>);
  static_assert(Kokkos::is_assignable_mem_space_v<Scratch, Scratch>);
  static_assert(!Kokkos::is_accessible_mem_space_v<void, Device>);
  static_assert(!Kokkos::is_accessible_mem_space_v<Host, Device>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Managed &>);

  static_assert(check_hip_memory_access<Kokkos::HIP>());
#ifdef KOKKOS_ENABLE_SERIAL
  static_assert(check_hip_memory_access<Kokkos::Serial>());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  static_assert(check_hip_memory_access<Kokkos::OpenMP>());
#endif
#ifdef KOKKOS_ENABLE_THREADS
  static_assert(check_hip_memory_access<Kokkos::Threads>());
#endif
#ifdef KOKKOS_ENABLE_HPX
  static_assert(check_hip_memory_access<Kokkos::Experimental::HPX>());
#endif
}

__global__ void test_abort() { Kokkos::abort("test_abort"); }

__global__ void test_hip_spaces_int_value(int *ptr) {
  if (*ptr == 42) {
    *ptr = 2 * 42;
  }
}

TEST(hip, space_access) {
  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                Kokkos::HostSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::HIPHostPinnedSpace>::assignable);

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                 Kokkos::HIPSpace>::assignable);

#if !defined(KOKKOS_IMPL_HIP_UNIFIED_MEMORY)
  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                 Kokkos::HIPSpace>::accessible);
#else
  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                Kokkos::HIPSpace>::accessible);
#endif

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                       Kokkos::HIPManagedSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::HIPManagedSpace>::accessible);

  //--------------------------------------

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                                Kokkos::HIPSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                       Kokkos::HIPHostPinnedSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                      Kokkos::HIPHostPinnedSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                       Kokkos::HostSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                       Kokkos::HostSpace>::accessible);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                      Kokkos::HIPManagedSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPSpace,
                                      Kokkos::HIPManagedSpace>::accessible);

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                      Kokkos::HIPHostPinnedSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                       Kokkos::HostSpace>::assignable);

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                Kokkos::HostSpace>::accessible);

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                 Kokkos::HIPSpace>::assignable);

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                                 Kokkos::HIPSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                       Kokkos::HIPManagedSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPHostPinnedSpace,
                                      Kokkos::HIPManagedSpace>::accessible);

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                      Kokkos::HIPManagedSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                       Kokkos::HostSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                       Kokkos::HostSpace>::accessible);

  static_assert(!Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                                 Kokkos::HIPSpace>::assignable);

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                                Kokkos::HIPSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                       Kokkos::HIPHostPinnedSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HIPManagedSpace,
                                      Kokkos::HIPHostPinnedSpace>::accessible);

  //--------------------------------------

  static_assert(
      !Kokkos::SpaceAccessibility<Kokkos::HIP, Kokkos::HostSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HIP, Kokkos::HIPSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HIP,
                                 Kokkos::HIPHostPinnedSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HIP,
                                 Kokkos::HIPManagedSpace>::accessible);

#if !defined(KOKKOS_IMPL_HIP_UNIFIED_MEMORY)
  static_assert(!Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                            Kokkos::HIPSpace>::accessible);
#else
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                           Kokkos::HIPSpace>::accessible);
#endif

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 Kokkos::HIPHostPinnedSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 Kokkos::HIPManagedSpace>::accessible);

#if !defined(KOKKOS_IMPL_HIP_UNIFIED_MEMORY)
  static_assert(
      std::is_same_v<Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::Space,
                     Kokkos::HostSpace>);
  static_assert(
      std::is_same_v<Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::device_type,
                     Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                                    Kokkos::HostSpace>>);
#else
  // Memory space stays the same as host can access HIPSpace
  static_assert(
      std::is_same_v<Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::Space,
                     Kokkos::HIPSpace>);
  static_assert(
      std::is_same_v<
          Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::device_type,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HIPSpace>>);
#endif

  static_assert(
      std::is_same_v<Kokkos::Impl::HostMirror<Kokkos::HIPManagedSpace>::Space,
                     Kokkos::HIPManagedSpace>);
  static_assert(std::is_same_v<
                Kokkos::Impl::HostMirror<Kokkos::HIPManagedSpace>::device_type,
                Kokkos::Device<Kokkos::HostSpace::execution_space,
                               Kokkos::HIPManagedSpace>>);

  static_assert(std::is_same_v<
                Kokkos::Impl::HostMirror<Kokkos::HIPHostPinnedSpace>::Space,
                Kokkos::HIPHostPinnedSpace>);
  static_assert(
      std::is_same_v<
          Kokkos::Impl::HostMirror<Kokkos::HIPHostPinnedSpace>::device_type,
          Kokkos::Device<Kokkos::HostSpace::execution_space,
                         Kokkos::HIPHostPinnedSpace>>);

  static_assert(Kokkos::SpaceAccessibility<
                Kokkos::Impl::HostMirror<Kokkos::HIPSpace>::device_type,
                Kokkos::HostSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<
          Kokkos::Impl::HostMirror<Kokkos::HIPHostPinnedSpace>::device_type,
          Kokkos::HostSpace>::accessible);

  static_assert(Kokkos::SpaceAccessibility<
                Kokkos::Impl::HostMirror<Kokkos::HIPManagedSpace>::device_type,
                Kokkos::HostSpace>::accessible);
}

template <class MemSpace, class ExecSpace>
struct TestViewHIPAccessible {
  enum { N = 1000 };

  using V = Kokkos::View<double *, MemSpace>;

  V m_base;

  struct TagInit {};
  struct TagTest {};

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagInit &, const int i) const { m_base[i] = i + 1; }

  KOKKOS_INLINE_FUNCTION
  void operator()(const TagTest &, const int i, long &error_count) const {
    if (m_base[i] != i + 1) ++error_count;
  }

  TestViewHIPAccessible() : m_base("base", N) {}

  static void run() {
    TestViewHIPAccessible self;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<typename MemSpace::execution_space, TagInit>(0, N),
        self);
    typename MemSpace::execution_space().fence();

    // Next access is a different execution space, must complete prior kernel.
    long error_count = -1;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecSpace, TagTest>(0, N), self,
                            error_count);
    EXPECT_EQ(error_count, 0);
  }
};

TEST(hip, impl_view_accessible) {
  TestViewHIPAccessible<Kokkos::HIPSpace, Kokkos::HIP>::run();

  TestViewHIPAccessible<Kokkos::HIPHostPinnedSpace, Kokkos::HIP>::run();
  TestViewHIPAccessible<Kokkos::HIPHostPinnedSpace,
                        Kokkos::HostSpace::execution_space>::run();

  TestViewHIPAccessible<Kokkos::HIPManagedSpace,
                        Kokkos::HostSpace::execution_space>::run();
  TestViewHIPAccessible<Kokkos::HIPManagedSpace, Kokkos::HIP>::run();
}

}  // namespace Test
