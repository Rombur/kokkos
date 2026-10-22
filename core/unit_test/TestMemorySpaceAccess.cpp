// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_EXPERIMENTAL_CXX20_MODULES
import kokkos.core;
import kokkos.core_impl;
#else
#include <Kokkos_Core.hpp>
#endif

namespace TestMemorySpaceAccess {

struct IncompleteExecution;
struct EarlyMemory {
  using memory_space    = EarlyMemory;
  using execution_space = IncompleteExecution;
};

// Backend space headers query self-assignment before defining execution spaces.
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
static_assert(
    Kokkos::Impl::MemorySpaceAccess<EarlyMemory, EarlyMemory>::assignable);
static_assert(
    Kokkos::Impl::MemorySpaceAccess<EarlyMemory, EarlyMemory>::accessible);
#endif

struct MemoryA;
struct MemoryB;
struct Scratch;

struct ExecutionA {
  using execution_space      = ExecutionA;
  using memory_space         = MemoryA;
  using scratch_memory_space = Scratch;
};

struct ExecutionB {
  using execution_space      = ExecutionB;
  using memory_space         = MemoryA;
  using scratch_memory_space = Scratch;
};

struct MemoryA {
  using memory_space    = MemoryA;
  using execution_space = ExecutionA;
};

struct MemoryB {
  using memory_space    = MemoryB;
  using execution_space = ExecutionB;
};

struct Scratch {
  using memory_space    = Scratch;
  using execution_space = ExecutionA;
};

}  // namespace TestMemorySpaceAccess

namespace Kokkos {

template <>
struct is_assignable_mem_space<TestMemorySpaceAccess::MemoryA,
                               TestMemorySpaceAccess::MemoryB>
    : std::true_type {};

template <>
struct is_accessible_mem_space<TestMemorySpaceAccess::ExecutionA,
                               TestMemorySpaceAccess::MemoryB>
    : std::true_type {};

template <>
struct is_accessible_mem_space<TestMemorySpaceAccess::ExecutionB,
                               TestMemorySpaceAccess::MemoryB>
    : std::false_type {};

}  // namespace Kokkos

namespace TestMemorySpaceAccess {

// These execution spaces share their default memory space, but their access
// to MemoryB differs. Only the new public traits define the relationships.
static_assert(Kokkos::is_accessible_mem_space_v<ExecutionA, MemoryB>);
static_assert(!Kokkos::is_accessible_mem_space_v<ExecutionB, MemoryB>);
static_assert(
    Kokkos::is_accessible_mem_space_v<const ExecutionA, volatile MemoryB>);
static_assert(
    Kokkos::is_assignable_mem_space_v<const MemoryA, volatile MemoryB>);

static_assert(Kokkos::SpaceAccessibility<ExecutionA, MemoryB>::accessible);
static_assert(!Kokkos::SpaceAccessibility<ExecutionB, MemoryB>::accessible);
static_assert(Kokkos::SpaceAccessibility<MemoryA, MemoryB>::assignable);
static_assert(!Kokkos::SpaceAccessibility<ExecutionA, MemoryB>::assignable);
static_assert(Kokkos::SpaceAccessibility<Kokkos::Device<ExecutionA, MemoryB>,
                                         MemoryA>::accessible);
static_assert(!Kokkos::SpaceAccessibility<Kokkos::Device<ExecutionA, MemoryA>,
                                          MemoryB>::assignable);
static_assert(
    std::is_same_v<Kokkos::SpaceAccessibility<ExecutionA, MemoryB>::space,
                   Kokkos::Device<ExecutionA, MemoryB>>);
static_assert(
    std::is_same_v<Kokkos::SpaceAccessibility<ExecutionB, MemoryB>::space,
                   ExecutionB>);

static_assert(!Kokkos::is_accessible_mem_space_v<ExecutionA, Scratch>);

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
// The old trait is independent of specializations of the new public traits.
static_assert(!Kokkos::Impl::MemorySpaceAccess<MemoryA, MemoryB>::assignable);
static_assert(!Kokkos::Impl::MemorySpaceAccess<MemoryA, MemoryB>::accessible);
static_assert(!Kokkos::Impl::MemorySpaceAccess<MemoryB, MemoryA>::accessible);

// Legacy scratch self-access stays true, even if execution access is false.
static_assert(Kokkos::Impl::MemorySpaceAccess<Scratch, Scratch>::accessible);
static_assert(!Kokkos::Impl::MemorySpaceAccess<Scratch, MemoryA>::accessible);

// Qualified legacy queries keep their original identity-only matching.
static_assert(
    !Kokkos::Impl::MemorySpaceAccess<const MemoryA, MemoryA>::assignable);
static_assert(
    !Kokkos::Impl::MemorySpaceAccess<const MemoryA, MemoryA>::accessible);
static_assert(
    Kokkos::Impl::MemorySpaceAccess<const MemoryA, const MemoryA>::assignable);
static_assert(
    Kokkos::Impl::MemorySpaceAccess<const MemoryA, const MemoryA>::accessible);
static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::AnonymousSpace,
                                              const MemoryB>::accessible);
static_assert(Kokkos::Impl::MemorySpaceAccess<
              const MemoryB, Kokkos::AnonymousSpace>::accessible);
#endif

}  // namespace TestMemorySpaceAccess
