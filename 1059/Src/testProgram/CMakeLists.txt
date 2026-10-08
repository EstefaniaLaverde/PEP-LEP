#source app
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/build)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/TestDataBig ${CMAKE_BINARY_DIR}/build/Debug/TestDataBig COPYONLY)
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/TestDataSmall ${CMAKE_BINARY_DIR}/build/Debug/TestDataSmall COPYONLY)
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/ArticleData ${CMAKE_BINARY_DIR}/build/Debug/ArticleData COPYONLY)
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/TestDataBig ${CMAKE_BINARY_DIR}/build/Release/TestDataBig COPYONLY)
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/TestDataSmall ${CMAKE_BINARY_DIR}/build/Release/TestDataSmall COPYONLY)
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/ArticleData ${CMAKE_BINARY_DIR}/build/Release/ArticleData COPYONLY)

else()
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/TestDataBig ${CMAKE_BINARY_DIR}/build/TestDataBig COPYONLY)
	configure_file( ${PROJECT_SOURCE_DIR}/testFiles/TestDataSmall ${CMAKE_BINARY_DIR}/build/TestDataSmall COPYONLY)
		configure_file( ${PROJECT_SOURCE_DIR}/testFiles/ArticleData ${CMAKE_BINARY_DIR}/build/ArticleData COPYONLY)
endif()


if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_executable(Test-UI Source.cpp)
target_link_libraries (Test-UI PRIVATE v1.3)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")

add_executable(test128_WeightDistribution Test128.cpp)
target_link_libraries (test128_WeightDistribution PRIVATE v1.3)
add_executable(testScalar_WeightDistribution TestScalar.cpp)
target_link_libraries (testScalar_WeightDistribution PRIVATE v1.3)


add_executable(test128_MinimumDistance Test128_MinimumDistance.cpp)
target_link_libraries (test128_MinimumDistance PRIVATE v1.3)
add_executable(testScalar_MinimumDistance TestScalar_MinimumDistance.cpp)  
target_link_libraries (testScalar_MinimumDistance PRIVATE v1.3)


add_executable(test128_SearchEqual Test128_SearchEqual.cpp)
target_link_libraries (test128_SearchEqual PRIVATE v1.3)
add_executable(testScalar_SearchEqual TestScalar_SearchEqual.cpp)
target_link_libraries (testScalar_SearchEqual PRIVATE v1.3)


add_executable(test128_SearchLessThan Test128_SearchLessThan.cpp)
target_link_libraries (test128_SearchLessThan PRIVATE v1.3)
add_executable(testScalar_SearchLessThan TestScalar_SearchLessThan.cpp)
target_link_libraries (testScalar_SearchLessThan PRIVATE v1.3)


add_executable(test128_CountEqual Test128_CountEqual.cpp)
target_link_libraries (test128_CountEqual PRIVATE v1.3)
add_executable(testScalar_CountEqual TestScalar_CountEqual.cpp)
target_link_libraries (testScalar_CountEqual PRIVATE v1.3)

add_executable(test128_CountLessThan Test128_CountLessThan.cpp)
target_link_libraries (test128_CountLessThan PRIVATE v1.3)
add_executable(testScalar_CountLessThan TestScalar_CountLessThan.cpp)
target_link_libraries (testScalar_CountLessThan PRIVATE v1.3)




else()
add_executable(test128_WeightDistribution Test128.cpp)
target_link_libraries (test128_WeightDistribution PRIVATE v1.3)
add_executable(test256_WeightDistribution Test256.cpp)
target_link_libraries (test256_WeightDistribution PRIVATE v1.3)
add_executable(test512_WeightDistribution Test512.cpp)
target_link_libraries (test512_WeightDistribution PRIVATE v1.3)
add_executable(testScalar_WeightDistribution TestScalar.cpp)
target_link_libraries (testScalar_WeightDistribution PRIVATE v1.3)



add_executable(test128_MinimumDistance Test128_MinimumDistance.cpp)
target_link_libraries (test128_MinimumDistance PRIVATE v1.3)
add_executable(test256_MinimumDistance Test256_MinimumDistance.cpp)
target_link_libraries (test256_MinimumDistance PRIVATE v1.3)
add_executable(test512_MinimumDistance Test512_MinimumDistance.cpp)
target_link_libraries (test512_MinimumDistance PRIVATE v1.3)
add_executable(testScalar_MinimumDistance TestScalar_MinimumDistance.cpp)  
target_link_libraries (testScalar_MinimumDistance PRIVATE v1.3)



add_executable(test128_SearchEqual Test128_SearchEqual.cpp)
target_link_libraries (test128_SearchEqual PRIVATE v1.3)
add_executable(test256_SearchEqual Test256_SearchEqual.cpp)
target_link_libraries (test256_SearchEqual PRIVATE v1.3)
add_executable(test512_SearchEqual Test512_SearchEqual.cpp)
target_link_libraries (test512_SearchEqual PRIVATE v1.3)
add_executable(testScalar_SearchEqual TestScalar_SearchEqual.cpp)
target_link_libraries (testScalar_SearchEqual PRIVATE v1.3)



add_executable(test128_SearchLessThan Test128_SearchLessThan.cpp)
target_link_libraries (test128_SearchLessThan PRIVATE v1.3)
add_executable(test256_SearchLessThan Test256_SearchLessThan.cpp)
target_link_libraries (test256_SearchLessThan PRIVATE v1.3)
add_executable(test512_SearchLessThan Test512_SearchLessThan.cpp)
target_link_libraries (test512_SearchLessThan PRIVATE v1.3)
add_executable(testScalar_SearchLessThan TestScalar_SearchLessThan.cpp)
target_link_libraries (testScalar_SearchLessThan PRIVATE v1.3)



add_executable(test128_CountEqual Test128_CountEqual.cpp)
target_link_libraries (test128_CountEqual PRIVATE v1.3)
add_executable(test256_CountEqual Test256_CountEqual.cpp)
target_link_libraries (test256_CountEqual PRIVATE v1.3)
add_executable(test512_CountEqual Test512_CountEqual.cpp)
target_link_libraries (test512_CountEqual PRIVATE v1.3)
add_executable(testScalar_CountEqual TestScalar_CountEqual.cpp)
target_link_libraries (testScalar_CountEqual PRIVATE v1.3)



add_executable(test128_CountLessThan Test128_CountLessThan.cpp)
target_link_libraries (test128_CountLessThan PRIVATE v1.3)
add_executable(test256_CountLessThan Test256_CountLessThan.cpp)
target_link_libraries (test256_CountLessThan PRIVATE v1.3)
add_executable(test512_CountLessThan Test512_CountLessThan.cpp)
target_link_libraries (test512_CountLessThan PRIVATE v1.3)
add_executable(testScalar_CountLessThan TestScalar_CountLessThan.cpp)
target_link_libraries (testScalar_CountLessThan PRIVATE v1.3)



endif()


#install(TARGETS test DESTINATION ${PROJECT_SOURCE_DIR}/ExampleFiles)