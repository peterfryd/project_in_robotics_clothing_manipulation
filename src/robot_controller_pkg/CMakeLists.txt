cmake_minimum_required(VERSION 3.8)
project(robot_controller_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_interfaces_pkg REQUIRED)
find_package(Eigen3 REQUIRED)

# find_library(RTDE_LIB rtde)
# find_path(RTDE_INCLUDE_DIR ur_rtde/rtde.h)

find_package(PkgConfig REQUIRED)

# # Arm controller
# add_executable(arm_controller src/arm_controller.cpp)
# ament_target_dependencies(arm_controller
#   rclcpp
#   sensor_msgs
#   custom_interfaces_pkg
# )
# target_include_directories(arm_controller PRIVATE ${RTDE_INCLUDE_DIR})
# target_link_libraries(arm_controller ${RTDE_LIB} ${MODBUS_LIBRARIES} Eigen3::Eigen)
# target_link_libraries(arm_controller ${MODBUS_LIBRARIES} Eigen3::Eigen)

# target_include_directories(gripper_controller PRIVATE ${RTDE_INCLUDE_DIR})
# target_link_libraries(gripper_controller ${RTDE_LIB} ${MODBUS_LIBRARIES})
# target_link_libraries(gripper_controller ${MODBUS_LIBRARIES})

# Fold point to point node
add_executable(fold_point_to_point src/fold_point_to_point.cpp)
ament_target_dependencies(fold_point_to_point
  rclcpp
  sensor_msgs
  custom_interfaces_pkg
)

# Install executables
install(TARGETS
  # arm_controller
  # gripper_controller
  fold_point_to_point
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
