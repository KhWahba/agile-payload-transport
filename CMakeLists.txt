
cmake_minimum_required(VERSION 3.5)
project(
  dynobench
  VERSION 0.1.0
  LANGUAGES CXX)

# --- MuJoCo root (portable, same style as pc-dbCBS) ---
set(MJ_ROOT "" CACHE PATH "Path to MuJoCo root (contains include/ and lib/)")

if(NOT MJ_ROOT)
  if(DEFINED ENV{MJ_ROOT})
    set(MJ_ROOT "$ENV{MJ_ROOT}")
  else()
    set(MJ_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../pc-dbCBS/deps/mujoco-3.3.6")
  endif()
endif()

message(STATUS "MJ_ROOT = ${MJ_ROOT}")

if(NOT EXISTS "${MJ_ROOT}/include/mujoco/mujoco.h")
  message(FATAL_ERROR "MuJoCo headers not found: ${MJ_ROOT}/include/mujoco/mujoco.h")
endif()

if(NOT EXISTS "${MJ_ROOT}/lib/libmujoco.so")
  message(FATAL_ERROR "MuJoCo library not found: ${MJ_ROOT}/lib/libmujoco.so")
endif()

# ONNX Runtime (CPU)
set(ONNXRUNTIME_ROOT "$ENV{HOME}/imitation-agile-payload-manipulation/deps/agile-payload-transport/deps/onnxruntime" CACHE PATH "ONNX Runtime root directory")

find_package(Boost REQUIRED COMPONENTS program_options serialization
                                       stacktrace_basic)
find_package(fcl REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(crocoddyl REQUIRED)
message(STATUS "CROCO_VERSION: ${CROCODDYL_VERSION}")
find_package(glfw3 CONFIG QUIET)
if(NOT glfw3_FOUND)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW REQUIRED glfw3) # uses glfw3.pc
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

option(BUILD_NMPC_PYTHON_BINDINGS "Build NMPC Python bindings" ON)

if(BUILD_NMPC_PYTHON_BINDINGS)
  set(PYBIND11_VENDOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pc-dbCBS/deps/dynoplan/dynobench/deps/pybind11")
  if(EXISTS "${PYBIND11_VENDOR_DIR}/CMakeLists.txt")
    add_subdirectory(${PYBIND11_VENDOR_DIR} ${CMAKE_BINARY_DIR}/pybind11_build EXCLUDE_FROM_ALL)
  else()
    find_package(pybind11 CONFIG REQUIRED)
  endif()
endif()


add_library(
  dynobench
  ./src/robot_models.cpp
  ./src/robot_models_base.cpp
  ./src/motions.cpp
  ./src/mujoco_quadrotors_payload.cpp
  ./src/mujoco_quadrotor.cpp
  ${SOURCE_n56})

add_library(dynobench::dynobench ALIAS dynobench)

set_property(TARGET dynobench PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(mujoco::mujoco SHARED IMPORTED)
set_target_properties(
  mujoco::mujoco PROPERTIES IMPORTED_LOCATION ${MJ_ROOT}/lib/libmujoco.so
                            INTERFACE_INCLUDE_DIRECTORIES ${MJ_ROOT}/include)

target_include_directories(
  dynobench
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${EIGEN3_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)


target_link_libraries(
  dynobench
  PUBLIC fcl
         yaml-cpp
         Boost::program_options
         Boost::serialization
         Boost::stacktrace_basic
         ${CMAKE_DL_LIBS}
         glfw
         mujoco::mujoco)


# ---- Policy ONNX Runtime wrapper ----
add_library(policy_onnx
  ./src/policy_onnx.cpp
)
set_property(TARGET policy_onnx PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  policy_onnx
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${ONNXRUNTIME_ROOT}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_directories(policy_onnx PUBLIC ${ONNXRUNTIME_ROOT}/lib)

target_link_libraries(
  policy_onnx
  PUBLIC Eigen3::Eigen
  PRIVATE onnxruntime
)


add_executable(nmpc_mujoco ./src/nmpc_mujoco.cpp)

add_library(nmpc_controller
  ./src/nmpc_controller.cpp
)

add_library(
  nmpc_core
  ./src/croco_models.cpp
  ./src/options.cpp
)

set_property(TARGET nmpc_core PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET nmpc_controller PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
  nmpc_controller
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(
  nmpc_core
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(
  nmpc_mujoco
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)


target_link_libraries(
nmpc_core
PUBLIC Eigen3::Eigen dynobench::dynobench
PRIVATE ${CROCODDYL_LIBRARIES} fcl yaml-cpp Boost::program_options
        Boost::serialization)

target_link_libraries(
nmpc_controller
PUBLIC policy_onnx
PRIVATE ${CROCODDYL_LIBRARIES}
        nmpc_core
        Eigen3::Eigen
        dynobench::dynobench
        fcl
        yaml-cpp
        Boost::program_options
        Boost::serialization)

target_link_libraries(
nmpc_mujoco
PRIVATE ${CROCODDYL_LIBRARIES}
        Eigen3::Eigen
        nmpc_controller)

# ---- Jacobian comparison test ----
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_jacobian_comparison.cpp")
  add_executable(test_jacobian_comparison tests/test_jacobian_comparison.cpp)
  target_include_directories(
    test_jacobian_comparison
    PUBLIC $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(test_jacobian_comparison PRIVATE dynobench::dynobench Eigen3::Eigen)
endif()

if(BUILD_NMPC_PYTHON_BINDINGS)
  pybind11_add_module(nmpc_controller_py MODULE ./src/py_nmpc_controller.cpp)
  target_include_directories(
    nmpc_controller_py
    PUBLIC $<INSTALL_INTERFACE:include>
           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           ${CROCODDYL_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(nmpc_controller_py PRIVATE nmpc_controller)
endif()

# ---- Standalone Crocoddyl + MuJoCo OCP demo (no project NMPC code) ----
add_executable(standalone_mj_croc_ocp ./standalone/standalone_mj_croc_ocp.cpp)
target_include_directories(
  standalone_mj_croc_ocp
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(
  standalone_mj_croc_ocp
  PRIVATE ${CROCODDYL_LIBRARIES}
          Eigen3::Eigen
          yaml-cpp
          mujoco::mujoco)
