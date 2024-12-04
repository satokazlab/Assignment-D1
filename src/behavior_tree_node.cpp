#include <rclcpp/rclcpp.hpp>  // ROS2のライブラリをインクルード
#include <std_msgs/msg/string.hpp>  // ROS2メッセージ型をインクルード

class BehaviorTreeNode : public rclcpp::Node
{
public:
    BehaviorTreeNode()
    : Node("behavior_tree_node")  // ノード名を指定
    {
        // タイマーを設定して、定期的にメッセージをパブリッシュ
        timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&BehaviorTreeNode::timer_callback, this)
        );

        // メッセージパブリッシャーを作成
        publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    }

private:
    void timer_callback()
    {
        // メッセージを作成
        auto message = std_msgs::msg::String();
        message.data = "Hello, ROS2!";

        // メッセージをパブリッシュ
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);  // ROS2を初期化
    rclcpp::spin(std::make_shared<BehaviorTreeNode>());  // ノードをスピン
    rclcpp::shutdown();  // ROS2をシャットダウン
    return 0;
}
