# Example 1
add_executable(example1 example1.cpp)
target_link_libraries(example1 PRIVATE meanshift)

# Example 2 (External Data)
add_executable(example2 example2.cpp)
target_link_libraries(example2 PRIVATE meanshift)

# Example 3 (Read from file but don't write)
add_executable(example3 example3.cpp)
target_link_libraries(example3 PRIVATE meanshift)

# Example 4 (Interactive Tool)
add_executable(example4 example4.cpp)
target_link_libraries(example4 PRIVATE meanshift)

# Example 5 (Batch Processing Loop)
add_executable(example5 example5.cpp)
target_link_libraries(example5 PRIVATE meanshift)

# Example 6 (Auto-Scanning Batch Processor)
add_executable(example6 example6.cpp)
target_link_libraries(example6 PRIVATE meanshift)
