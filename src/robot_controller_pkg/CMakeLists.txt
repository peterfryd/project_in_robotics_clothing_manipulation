cmake_minimum_required(VERSION 3.8)
project(robot_controller_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED) 
find_package(custom_interfaces_pkg REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PkgConfig REQUIRED)


add_executable(fold_point_to_point src/fold_point_to_point.cpp)
ament_target_dependencies(fold_point_to_point
  rclcpp
  sensor_msgs
  custom_interfaces_pkg
  std_srvs
)

install(TARGETS
  fold_point_to_point
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
