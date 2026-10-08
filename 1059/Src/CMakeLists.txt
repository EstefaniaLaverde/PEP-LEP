#project 
cmake_minimum_required(VERSION 3.16)

# set the project name and version

project(LinCodeWeightInv VERSION 1.3 LANGUAGES CXX )

if(NOT CMAKE_CONFIGURATION_TYPES)
	if("${CMAKE_BUILD_TYPE}" STREQUAL "")
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build Conf" FORCE)
	endif()
endif()

#testProgram contains the main testing programs 
#src contains the cpp files 

add_subdirectory(src)
add_subdirectory(testProgram)

