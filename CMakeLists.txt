
cmake_minimum_required(VERSION 3.5)
project(
  dynobench
  VERSION 0.1.0
  LANGUAGES CXX)

set(MJ_ROOT "$ENV{HOME}/inria/phdv2025-khaled_wahba/src/deps/mujoco-3.3.3")

# find_package(pinocchio REQUIRED)
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



add_executable(nmpc_mujoco ./src/nmpc_mujoco.cpp)


add_library(
  optimization
  ./src/ocp.cpp ./src/options.cpp
  ./src/croco_models.cpp ./src/generate_ocp.cpp
  ./src/opt_simulate_mujoco.cpp)


target_include_directories(
  optimization
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         ${CROCODDYL_INCLUDE_DIRS}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_include_directories(
  nmpc_mujoco
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)


target_link_libraries(
optimization
PUBLIC Eigen3::Eigen dynobench::dynobench
PRIVATE ${CROCODDYL_LIBRARIES} fcl yaml-cpp Boost::program_options
        Boost::serialization libipopt.so)

target_link_libraries(
nmpc_mujoco
PRIVATE ${CROCODDYL_LIBRARIES}
        optimization
        Eigen3::Eigen
        dynobench::dynobench
        fcl
        yaml-cpp
        Boost::program_options
        Boost::serialization)
