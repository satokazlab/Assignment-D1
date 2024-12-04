#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <cstdlib>

class SearchForPaperNode : public rclcpp::Node
{
public:
    SearchForPaperNode() : Node("cpp_subscriber_node")
    {
        // Pythonノードをバックグラウンドで起動
        std::system("ros2 run oakd_chan_package search_for_paper_node &");
        RCLCPP_INFO(this->get_logger(), "Started Python Search For Paper Node.");

        // サブスクライバーを作成
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "topic", 10, std::bind(&SearchForPaperNode::topic_callback, this, std::placeholders::_1));
    }

private:
    void topic_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received: '%s'", msg->data.c_str());
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SearchForPaperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
