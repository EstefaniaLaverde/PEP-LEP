#sources for library

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/lib_neon.h" "${PROJECT_SOURCE_DIR}/include/LinCodeWeightInv.h"  "${PROJECT_SOURCE_DIR}/include/ReadWrite.h" "${PROJECT_SOURCE_DIR}/include/Polynomials.h" "${PROJECT_SOURCE_DIR}/include/Data.h" "${PROJECT_SOURCE_DIR}/include/testDriver.h")
set(SOURCE_LIST_LIB "LinCodeWeightInv.cpp" "Polynomials.cpp"  "ReadWrite.cpp" "lib_neon.cpp"  "Data.cpp" "testDriver.cpp")
else()
set(HEADER_LIST "${PROJECT_SOURCE_DIR}/include/lib128.h" "${PROJECT_SOURCE_DIR}/include/lib256.h" "${PROJECT_SOURCE_DIR}/include/lib512.h" "${PROJECT_SOURCE_DIR}/include/LinCodeWeightInv.h"  "${PROJECT_SOURCE_DIR}/include/ReadWrite.h" "${PROJECT_SOURCE_DIR}/include/Polynomials.h" "${PROJECT_SOURCE_DIR}/include/Data.h" "${PROJECT_SOURCE_DIR}/include/testDriver.h")
set(SOURCE_LIST_LIB "LinCodeWeightInv.cpp" "Polynomials.cpp"  "ReadWrite.cpp" "lib128.cpp" "lib256.cpp" "lib512.cpp" "Data.cpp" "testDriver.cpp")
endif()

add_compile_definitions(FORCE_INSTR=0)
add_compile_definitions(AVX512POCNT_MANUAL=1)

add_library(v1.3 STATIC "${SOURCE_LIST_LIB}" "${HEADER_LIST}")
set_property(TARGET v1.3 PROPERTY CXX_STANDARD 17)


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native ")
endif()	
#-mavx512vpopcntdq

target_include_directories (v1.3 PUBLIC ../include)

install(TARGETS v1.3 DESTINATION ${PROJECT_SOURCE_DIR}/lib)


#to add header list to the project in IDE
source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES "${HEADER_LIST}")

