cmake_minimum_required(VERSION 3.18)
project(torchturbo_ascendc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Torch REQUIRED)
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
function(torchturbo_ascendc_library target kernel)
target_include_directories(${target} PRIVATE
    ${TORCH_NPU_PATH}/include ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_directories(${target} PRIVATE
    ${TORCH_NPU_PATH}/lib ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(${target} PRIVATE ${kernel} ${TORCH_LIBRARIES} torch_npu ascendcl)
target_link_options(${target} PRIVATE -Wl,--no-undefined)
set_target_properties(${target} PROPERTIES
    INSTALL_RPATH "${TORCH_NPU_PATH}/lib;${TORCH_INSTALL_PREFIX}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
install(TARGETS ${target} LIBRARY DESTINATION .)
endfunction()

add_subdirectory(gated_activation)
