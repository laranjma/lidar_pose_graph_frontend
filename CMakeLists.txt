cmake_minimum_required(VERSION 3.8)
project(lidar_pose_graph_frontend)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# Avoid broken GTSAM CMake exports by resolving headers/libs directly.
find_path(GTSAM_INCLUDE_DIR
  NAMES gtsam/geometry/Pose2.h
)
find_library(GTSAM_LIBRARY
  NAMES gtsam
)
if(NOT GTSAM_INCLUDE_DIR OR NOT GTSAM_LIBRARY)
  message(FATAL_ERROR "Could not find GTSAM include dir/library on this system.")
endif()

add_executable(pose_graph_node src/pose_graph_node.cpp)
ament_target_dependencies(
  pose_graph_node
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
)
target_link_libraries(pose_graph_node Eigen3::Eigen)
target_link_libraries(pose_graph_node TBB::tbb)
target_include_directories(pose_graph_node PUBLIC ${GTSAM_INCLUDE_DIR})
target_link_libraries(pose_graph_node ${GTSAM_LIBRARY})

install(TARGETS pose_graph_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
