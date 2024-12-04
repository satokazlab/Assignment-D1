#include <filesystem>

#include "behaviortree_cpp_v3/loggers/bt_file_logger.h"
#include "behaviortree_cpp_v3/loggers/bt_cout_logger.h"
#include "behaviortree_cpp_v3/loggers/bt_zmq_publisher.h"

//------------------------------------------------------------------------------------
//---------------------------- Include Thrid Party -----------------------------------
//------------------------------------------------------------------------------------
// #include "tb3_behavior_tree/read_laser_condition.hpp"
// #include "tb3_behavior_tree/detect_camera_action.hpp"
// #include "tb3_behavior_tree/move_robot_action.hpp"
// #include "tb3_behavior_tree/rotating_robot_action.hpp"
// #include "tb3_behavior_tree/stop_robot_action.hpp"

#include <rclcpp/rclcpp.hpp>  // 必要なヘッダをインクルード
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <behaviortree_cpp_v3/basic_types.h>
#include <behaviortree_cpp_v3/tree_node.h>


using namespace std::chrono_literals;
using std::chrono::milliseconds;
using std::placeholders::_1;
std::atomic_bool switchActive{true};

using namespace BT;


int main(int argc, char **argv) {

  std::filesystem::path ros_ws_path = std::filesystem::current_path();
  std::string xml_file_name = "/src/oakd_chan/config/paper_tree.xml";
  std::string xml_path = ros_ws_path.string() + xml_file_name;

  rclcpp::init(argc, argv);

 // We use the BehaviorTreeFactory to register our custom nodes
  BehaviorTreeFactory factory;

  //Node registration process
  factory.registerNodeType<SearchForPaperNode>("SearchForPaperNode");
//   factory.registerNodeType<CameraDetected>("CameraDetected");
//   factory.registerNodeType<ReadingLaser>("ReadingLaser");
//   factory.registerNodeType<Rotating>("Rotating");
//   factory.registerNodeType<Stop>("Stop");

  // we incorporated the BT (XML format)
  auto tree = factory.createTreeFromFile(xml_path);

  // PublisherZMQ(const BT::Tree& tree, unsigned max_msg_per_second = 25,
  //               unsigned publisher_port = 1666, unsigned server_port = 1667);
  unsigned publisher_port = 1666;
  unsigned server_port = 1667;
  BT::PublisherZMQ publisher_zmq(tree, 25, publisher_port, server_port);

  BT::NodeStatus status = BT::NodeStatus::FAILURE;
  BT::NodeConfiguration con = {};

  //definiion of smart pointers to
  // auto lc_listener = std::make_shared<ReadingLaser>("lc_listener", con);
  // auto lc_odom = std::make_shared<Rotating>("lc_odom", con);
  // auto lc_camera = std::make_shared<CameraDetected>("lc_camera", con);
  // auto lc_stop = std::make_shared<Stop>("lc_stop", con);

  // console log
  BT::StdCoutLogger logger_cout(tree);
  // for logging purposes. Details later
  FileLogger logger_file(tree, "src/tb3_behavior_tree/log/tb3_bts_trace.fbl");
  // we spin ROS nodes
  while (rclcpp::ok() && status == BT::NodeStatus::FAILURE) {
    // rclcpp::spin_some(lc_odom);
    // rclcpp::spin_some(lc_listener);
    // rclcpp::spin_some(lc_camera);
    // rclcpp::spin_some(lc_stop);
    //we check the status of node
    status = tree.tickRoot();

    // Groot 4.X
    // status = tree.tickOnce();
    // status = tree.tickWhileRunning();

    tree.sleep(std::chrono::milliseconds(200));
  }

  return 0;
}
