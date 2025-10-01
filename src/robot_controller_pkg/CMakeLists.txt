cmake_minimum_required(VERSION 3.8)
project(robot_controller_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(custom_interfaces_pkg REQUIRED)

find_library(RTDE_LIB rtde)
find_path(RTDE_INCLUDE_DIR ur_rtde/rtde.h)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MODBUS REQUIRED libmodbus)
include_directories(${MODBUS_INCLUDE_DIRS})

# Arm controller
add_executable(arm_controller src/arm_controller.cpp)
ament_target_dependencies(arm_controller
  rclcpp
  sensor_msgs
  custom_interfaces_pkg
)
target_include_directories(arm_controller PRIVATE ${RTDE_INCLUDE_DIR})
target_link_libraries(arm_controller ${RTDE_LIB} ${MODBUS_LIBRARIES})

# Gripper controller
add_executable(gripper_controller src/gripper_controller.cpp)
ament_target_dependencies(gripper_controller
  rclcpp
  sensor_msgs
  custom_interfaces_pkg
)
target_include_directories(gripper_controller PRIVATE ${RTDE_INCLUDE_DIR})
target_link_libraries(gripper_controller ${RTDE_LIB} ${MODBUS_LIBRARIES})

# Install executables
install(TARGETS
  arm_controller
  gripper_controller
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
