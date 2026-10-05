# Tests
add_executable(test_meanshift test_meanshift.cpp)
target_link_libraries(test_meanshift PRIVATE meanshift)
add_test(NAME MeanShiftTests COMMAND test_meanshift)
