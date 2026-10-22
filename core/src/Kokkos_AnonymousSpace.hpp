// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif
#ifndef KOKKOS_ANONYMOUSSPACE_HPP
#define KOKKOS_ANONYMOUSSPACE_HPP

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <cstddef>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

class AnonymousSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space    = AnonymousSpace;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using size_type       = size_t;
  using index_type      = std::make_signed_t<size_type>;

  //! This memory space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  /**\brief  Default memory space instance */
  AnonymousSpace() = default;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char *name() { return "Anonymous"; }
};

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class OtherSpace>
struct is_assignable_mem_space_helper<AnonymousSpace, OtherSpace, true>
    : std::true_type {};

template <class OtherSpace>
struct is_assignable_mem_space_helper<OtherSpace, AnonymousSpace, true>
    : std::true_type {};

template <>
struct is_assignable_mem_space_helper<AnonymousSpace, AnonymousSpace, true>
    : std::true_type {};

template <class ExecutionSpace>
struct is_accessible_mem_space_helper<ExecutionSpace, AnonymousSpace, true>
    : std::true_type {};

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE_5
template <typename OtherSpace>
struct MemorySpaceAccess<Kokkos::AnonymousSpace, OtherSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
};

template <typename OtherSpace>
struct MemorySpaceAccess<OtherSpace, Kokkos::AnonymousSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
};

template <>
struct MemorySpaceAccess<Kokkos::AnonymousSpace, Kokkos::AnonymousSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
};
#endif

}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_ANONYMOUSSPACE_HPP
