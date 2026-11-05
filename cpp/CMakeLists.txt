cmake_minimum_required(VERSION 3.16)
project(bup_engine VERSION 2.0.0 LANGUAGES CXX)

# dx_app coding standard: C++14, RAII, no raw new/delete.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# DEEPX runtime: /usr/local/lib/cmake/dxrt (libdxrt-bin package or a dx_rt source install)
find_package(dxrt REQUIRED)
# DXRT >= 3.4 exports dxrt::dxrt; DXRT <= 3.3 (dx_rt source install) only sets DXRT_LIBRARY.
if(NOT TARGET dxrt::dxrt)
  find_path(DXRT_INCLUDE_DIR NAMES dxrt/dxrt_api.h HINTS ${CMAKE_INSTALL_PREFIX}/include /usr/local/include)
  if(NOT DXRT_LIBRARY OR NOT DXRT_INCLUDE_DIR)
    message(FATAL_ERROR "DXRT found but its library/headers could not be located (DXRT_LIBRARY=${DXRT_LIBRARY}, DXRT_INCLUDE_DIR=${DXRT_INCLUDE_DIR})")
  endif()
  add_library(dxrt::dxrt UNKNOWN IMPORTED)
  set_target_properties(dxrt::dxrt PROPERTIES
    IMPORTED_LOCATION "${DXRT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${DXRT_INCLUDE_DIR}")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

add_library(bup_core STATIC
  src/common.cpp
  src/stream_url.cpp
  src/tracker.cpp
  src/byte_tracker.cpp
  src/face_align.cpp
  src/factories.cpp
  src/face_engine.cpp
  src/capture.cpp
  src/overlay.cpp
  src/session.cpp
  src/system_monitor.cpp)
target_include_directories(bup_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/dxapp)   # vendored dx_app "common/..." headers
target_link_libraries(bup_core PUBLIC dxrt::dxrt ${OpenCV_LIBS} PkgConfig::GST Threads::Threads)
target_compile_definitions(bup_core PUBLIC BUP_ENGINE_VERSION="${PROJECT_VERSION}")
target_compile_options(bup_core PRIVATE -Wall -Wextra)

# Python module used by app.py
pybind11_add_module(bup_engine src/bindings.cpp)
target_link_libraries(bup_engine PRIVATE bup_core)

# Operator CLI: detect / bench / stream (camera debugging without Flask)
add_executable(bupctl tools/bupctl.cpp)
target_link_libraries(bupctl PRIVATE bup_core)
