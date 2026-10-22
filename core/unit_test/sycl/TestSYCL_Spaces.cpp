// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
#else
#include <Kokkos_Core.hpp>
#endif
#include <TestSYCL_Category.hpp>

namespace Test {

template <class ExecutionSpace>
constexpr bool check_sycl_memory_access() {
  constexpr bool is_sycl = std::is_same_v<ExecutionSpace, Kokkos::SYCL>;
  constexpr bool is_host =
      std::is_same_v<typename ExecutionSpace::memory_space, Kokkos::HostSpace>;

  static_assert(Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                                  Kokkos::SYCLDeviceUSMSpace> ==
                is_sycl);
  static_assert(Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                                  Kokkos::SYCLSharedUSMSpace> ==
                (is_sycl || is_host));
  static_assert(Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                                  Kokkos::SYCLHostUSMSpace> ==
                (is_sycl || is_host));
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecutionSpace, Kokkos::HostSpace> ==
      is_host);
  static_assert(
      Kokkos::is_accessible_mem_space_v<ExecutionSpace,
                                        Kokkos::SYCL::scratch_memory_space> ==
      is_sycl);
  return true;
}

TEST(sycl, memory_space_traits) {
  using Host    = Kokkos::HostSpace;
  using Device  = Kokkos::SYCLDeviceUSMSpace;
  using Shared  = Kokkos::SYCLSharedUSMSpace;
  using HostUSM = Kokkos::SYCLHostUSMSpace;

  static_assert(Kokkos::is_assignable_mem_space_v<Host, Host>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Host, Device>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Host, Shared>);
  static_assert(Kokkos::is_assignable_mem_space_v<Host, HostUSM>);

  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Host>);
  static_assert(Kokkos::is_assignable_mem_space_v<Device, Device>);
  static_assert(Kokkos::is_assignable_mem_space_v<Device, Shared>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Device, HostUSM>);

  static_assert(!Kokkos::is_assignable_mem_space_v<Shared, Host>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Shared, Device>);
  static_assert(Kokkos::is_assignable_mem_space_v<Shared, Shared>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Shared, HostUSM>);

  static_assert(!Kokkos::is_assignable_mem_space_v<HostUSM, Host>);
  static_assert(!Kokkos::is_assignable_mem_space_v<HostUSM, Device>);
  static_assert(!Kokkos::is_assignable_mem_space_v<HostUSM, Shared>);
  static_assert(Kokkos::is_assignable_mem_space_v<HostUSM, HostUSM>);

  using Scratch = Kokkos::SYCL::scratch_memory_space;
  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Scratch>);
  static_assert(Kokkos::is_assignable_mem_space_v<Scratch, Scratch>);
  static_assert(!Kokkos::is_accessible_mem_space_v<void, Device>);
  static_assert(!Kokkos::is_accessible_mem_space_v<Host, Device>);
  static_assert(!Kokkos::is_assignable_mem_space_v<Device, Shared &>);

  static_assert(check_sycl_memory_access<Kokkos::SYCL>());
#ifdef KOKKOS_ENABLE_SERIAL
  static_assert(check_sycl_memory_access<Kokkos::Serial>());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  static_assert(check_sycl_memory_access<Kokkos::OpenMP>());
#endif
#ifdef KOKKOS_ENABLE_THREADS
  static_assert(check_sycl_memory_access<Kokkos::Threads>());
#endif
#ifdef KOKKOS_ENABLE_HPX
  static_assert(check_sycl_memory_access<Kokkos::Experimental::HPX>());
#endif
}

TEST(sycl, space_access) {
  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                                Kokkos::HostSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::SYCLHostUSMSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                       Kokkos::SYCLDeviceUSMSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                       Kokkos::SYCLDeviceUSMSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                       Kokkos::SYCLSharedUSMSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::SYCLSharedUSMSpace>::accessible);

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLDeviceUSMSpace,
                                      Kokkos::SYCLDeviceUSMSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLDeviceUSMSpace,
                                      Kokkos::SYCLSharedUSMSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLDeviceUSMSpace,
                                       Kokkos::SYCLHostUSMSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLDeviceUSMSpace,
                                      Kokkos::SYCLHostUSMSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLDeviceUSMSpace,
                                       Kokkos::HostSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLDeviceUSMSpace,
                                       Kokkos::HostSpace>::accessible);

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                      Kokkos::SYCLSharedUSMSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                       Kokkos::SYCLDeviceUSMSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                      Kokkos::SYCLDeviceUSMSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                       Kokkos::HostSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                       Kokkos::HostSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                       Kokkos::SYCLHostUSMSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLSharedUSMSpace,
                                      Kokkos::SYCLHostUSMSpace>::accessible);

  //--------------------------------------

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                      Kokkos::SYCLHostUSMSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                       Kokkos::HostSpace>::assignable);

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                                Kokkos::HostSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                       Kokkos::SYCLDeviceUSMSpace>::assignable);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                       Kokkos::SYCLDeviceUSMSpace>::accessible);

  static_assert(
      !Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                       Kokkos::SYCLSharedUSMSpace>::assignable);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                      Kokkos::SYCLSharedUSMSpace>::accessible);

  //--------------------------------------

  static_assert(
      !Kokkos::SpaceAccessibility<Kokkos::SYCL, Kokkos::HostSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::SYCL,
                                 Kokkos::SYCLDeviceUSMSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::SYCL,
                                 Kokkos::SYCLSharedUSMSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::SYCL,
                                 Kokkos::SYCLHostUSMSpace>::accessible);

  static_assert(
      !Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                  Kokkos::SYCLDeviceUSMSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 Kokkos::SYCLSharedUSMSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 Kokkos::SYCLHostUSMSpace>::accessible);

  static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::SYCLHostUSMSpace,
                                                Kokkos::HostSpace>::accessible);

  static_assert(
      Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                      Kokkos::SYCLHostUSMSpace>::accessible);

  static_assert(std::is_same_v<
                Kokkos::Impl::HostMirror<Kokkos::SYCLDeviceUSMSpace>::Space,
                Kokkos::HostSpace>);
  static_assert(
      std::is_same_v<
          Kokkos::Impl::HostMirror<Kokkos::SYCLSharedUSMSpace>::device_type,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         Kokkos::SYCLSharedUSMSpace>>);

  static_assert(
      std::is_same_v<Kokkos::Impl::HostMirror<Kokkos::SYCLHostUSMSpace>::Space,
                     Kokkos::SYCLHostUSMSpace>);
  static_assert(std::is_same_v<
                Kokkos::Impl::HostMirror<Kokkos::SYCLHostUSMSpace>::device_type,
                Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                               Kokkos::SYCLHostUSMSpace>>);

  static_assert(std::is_same_v<
                Kokkos::Impl::HostMirror<Kokkos::SYCLSharedUSMSpace>::Space,
                Kokkos::SYCLSharedUSMSpace>);
  static_assert(
      std::is_same_v<
          Kokkos::Impl::HostMirror<Kokkos::SYCLSharedUSMSpace>::device_type,
          Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                         Kokkos::SYCLSharedUSMSpace>>);

  static_assert(
      Kokkos::SpaceAccessibility<
          Kokkos::Impl::HostMirror<Kokkos::SYCLDeviceUSMSpace>::device_type,
          Kokkos::HostSpace>::accessible);

  static_assert(
      Kokkos::SpaceAccessibility<
          Kokkos::Impl::HostMirror<Kokkos::SYCLSharedUSMSpace>::device_type,
          Kokkos::HostSpace>::accessible);

  static_assert(Kokkos::SpaceAccessibility<
                Kokkos::Impl::HostMirror<Kokkos::SYCLHostUSMSpace>::device_type,
                Kokkos::HostSpace>::accessible);
}

TEST(sycl, uvm) {
  int *uvm_ptr =
      static_cast<int *>(Kokkos::kokkos_malloc<Kokkos::SYCLSharedUSMSpace>(
          "uvm_ptr", sizeof(int)));

  *uvm_ptr = 42;

  Kokkos::SYCL().fence();
  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::SYCL>(0, 1), KOKKOS_LAMBDA(int) {
        if (*uvm_ptr == 42) {
          *uvm_ptr = 2 * 42;
        }
      });
  Kokkos::SYCL().fence();

  EXPECT_EQ(*uvm_ptr, int(2 * 42));

  Kokkos::kokkos_free<Kokkos::SYCLSharedUSMSpace>(uvm_ptr);
}

template <class MemSpace, class ExecSpace>
struct TestViewSYCLAccessible {
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

  TestViewSYCLAccessible() : m_base("base", N) {}

  static void run() {
    TestViewSYCLAccessible self;
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

TEST(sycl, impl_view_accessible) {
  TestViewSYCLAccessible<Kokkos::SYCLDeviceUSMSpace, Kokkos::SYCL>::run();

  TestViewSYCLAccessible<Kokkos::SYCLSharedUSMSpace, Kokkos::SYCL>::run();
  TestViewSYCLAccessible<Kokkos::SYCLSharedUSMSpace,
                         Kokkos::HostSpace::execution_space>::run();

  TestViewSYCLAccessible<Kokkos::SYCLHostUSMSpace, Kokkos::SYCL>::run();
  TestViewSYCLAccessible<Kokkos::SYCLHostUSMSpace,
                         Kokkos::HostSpace::execution_space>::run();
}

}  // namespace Test
