cmake_minimum_required(VERSION 3.8)
project(system_integrator_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(custom_interfaces_pkg REQUIRED)
find_package(OpenCV REQUIRED)

# Executable
add_executable(main src/main.cpp)
add_executable(main2 src/main2.cpp)

ament_target_dependencies(main
  rclcpp
  sensor_msgs
  std_srvs
  custom_interfaces_pkg
  OpenCV
)

ament_target_dependencies(main2
  rclcpp
  sensor_msgs
  std_srvs
  custom_interfaces_pkg
  OpenCV
)

install(TARGETS main
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS main2
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
