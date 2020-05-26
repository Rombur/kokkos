KOKKOS_CFG_DEPENDS(COMPILER_ID NONE)

SET(KOKKOS_CXX_COMPILER ${CMAKE_CXX_COMPILER})
SET(KOKKOS_CXX_COMPILER_ID ${CMAKE_CXX_COMPILER_ID})
SET(KOKKOS_CXX_COMPILER_VERSION ${CMAKE_CXX_COMPILER_VERSION})

IF(Kokkos_ENABLE_CUDA)
# Check if the compiler is nvcc (which really means nvcc_wrapper).
EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                OUTPUT_VARIABLE INTERNAL_COMPILER_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)

STRING(REPLACE "\n" " - " INTERNAL_COMPILER_VERSION_ONE_LINE ${INTERNAL_COMPILER_VERSION} )

STRING(FIND ${INTERNAL_COMPILER_VERSION_ONE_LINE} "nvcc" INTERNAL_COMPILER_VERSION_CONTAINS_NVCC)


STRING(REGEX REPLACE "^ +" ""
       INTERNAL_HAVE_COMPILER_NVCC "${INTERNAL_HAVE_COMPILER_NVCC}")
IF(${INTERNAL_COMPILER_VERSION_CONTAINS_NVCC} GREATER -1)
  SET(INTERNAL_HAVE_COMPILER_NVCC true)
ELSE()
  SET(INTERNAL_HAVE_COMPILER_NVCC false)
ENDIF()
ENDIF()

IF(INTERNAL_HAVE_COMPILER_NVCC)
  # SET the compiler id to nvcc.  We use the value used by CMake 3.8.
  SET(KOKKOS_CXX_COMPILER_ID NVIDIA CACHE STRING INTERNAL FORCE)

  STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+$"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_COMPILER_VERSION_ONE_LINE})
  SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  MESSAGE(STATUS "Compiler Version: ${KOKKOS_CXX_COMPILER_VERSION}")
ENDIF()


IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  # The Cray compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep Cray
                  COMMAND wc -l
                  OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  IF (INTERNAL_HAVE_CRAY_COMPILER) #not actually Clang
    SET(KOKKOS_CLANG_IS_CRAY TRUE)
  ENDIF()
ENDIF()

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Cray OR KOKKOS_CLANG_IS_CRAY)
  # SET Cray's compiler version.
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  OUTPUT_VARIABLE INTERNAL_CXX_COMPILER_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  STRING(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+"
         TEMP_CXX_COMPILER_VERSION ${INTERNAL_CXX_COMPILER_VERSION})
  IF (KOKKOS_CLANG_IS_CRAY)
    SET(KOKKOS_CLANG_CRAY_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION})
  ELSE()
    SET(KOKKOS_CXX_COMPILER_VERSION ${TEMP_CXX_COMPILER_VERSION} CACHE STRING INTERNAL FORCE)
  ENDIF()
ENDIF()


# check for HCC vs HIP-Clang
EXECUTE_PROCESS(COMMAND "${CMAKE_CXX_COMPILER} --version --amdgpu-target=gfx906"
                COMMAND grep "HCC clang"
                COMMAND wc -l
                OUTPUT_VARIABLE INTERNAL_HAVE_COMPILER_HCC
                OUTPUT_STRIP_TRAILING_WHITESPACE)

IF(INTERNAL_HAVE_COMPILER_HCC)
  # SET the compiler id to hcc.  We use the value used by CMake 3.8.
  SET(KOKKOS_CXX_COMPILER_ID HCC CACHE STRING INTERNAL FORCE)
ENDIF()


# Enforce the minimum compilers supported by Kokkos.
SET(KOKKOS_MESSAGE_TEXT "Compiler not supported by Kokkos.  Required compiler versions:")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Clang      3.5.2 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    GCC        4.8.4 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    Intel     15.0.2 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    NVCC      9.0.69 or higher")
SET(KOKKOS_MESSAGE_TEXT "${KOKKOS_MESSAGE_TEXT}\n    PGI         17.1 or higher\n")

IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 3.5.2)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL GNU)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 4.8.4)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 15.0.2)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 9.0.69)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
  SET(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Kokkos turns off CXX extensions" FORCE)
ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL PGI)
  IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 17.1)
    MESSAGE(FATAL_ERROR "${KOKKOS_MESSAGE_TEXT}")
  ENDIF()
ENDIF()

STRING(REPLACE "." ";" VERSION_LIST ${KOKKOS_CXX_COMPILER_VERSION})
LIST(GET VERSION_LIST 0 KOKKOS_COMPILER_VERSION_MAJOR)
LIST(GET VERSION_LIST 1 KOKKOS_COMPILER_VERSION_MINOR)
LIST(GET VERSION_LIST 2 KOKKOS_COMPILER_VERSION_PATCH)
