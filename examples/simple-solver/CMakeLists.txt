add_executable(simple-solver simple-solver.cpp)
target_link_libraries(simple-solver ginkgo)
target_include_directories(simple-solver PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
