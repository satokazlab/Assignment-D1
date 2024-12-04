cmake_minimum_required(VERSION 3.5)
project(oakd_chan)

# C++標準を設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# パッケージを探す
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)  # OpenCVを使用
find_package(depthai CONFIG REQUIRED)  # DepthAI SDKを使用

# 実行可能ファイルの指定
add_executable(behavior_tree_node src/behavior_tree_node.cpp)

# ターゲットにインクルードディレクトリを指定
target_include_directories(behavior_tree_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# 必要なライブラリとリンク
target_link_libraries(behavior_tree_node
  ${OpenCV_LIBS}
)

# 必要な依存パッケージを指定
ament_target_dependencies(behavior_tree_node
  rclcpp
  std_msgs
  sensor_msgs
)

# インストール設定
install(TARGETS behavior_tree_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
