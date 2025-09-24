cmake_minimum_required(VERSION 3.8)
project(robot_controller_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- NEW: find the RTDE library ---
find_library(RTDE_LIB rtde)                     # finds /usr/lib*/librtde.so
find_path(RTDE_INCLUDE_DIR ur_rtde/rtde.h)      # finds /usr/include/ur_rtde/rtde.h

find_package(PkgConfig REQUIRED)
pkg_check_modules(MODBUS REQUIRED libmodbus)
include_directories(${MODBUS_INCLUDE_DIRS})


# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RobotCmd.msg"
  "srv/RobotCmd.srv"
)

ament_export_dependencies(rosidl_default_runtime)

# Get typesupport target for linking
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Add node executable
add_executable(robot_controller src/robot_controller.cpp)
ament_target_dependencies(robot_controller rclcpp)

# --- NEW: include RTDE headers and link the library ---
target_include_directories(robot_controller PRIVATE ${RTDE_INCLUDE_DIR})
target_link_libraries(robot_controller ${cpp_typesupport_target} ${RTDE_LIB})
target_link_libraries(robot_controller ${MODBUS_LIBRARIES})

install(TARGETS robot_controller DESTINATION lib/${PROJECT_NAME})

ament_package()
