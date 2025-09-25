#include <memory>
#include <mutex>
#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "custom_interfaces_pkg/srv/inference.hpp"
#include "custom_interfaces_pkg/srv/robot_cmd.hpp"

using namespace std::chrono_literals;

class SystemIntegration : public rclcpp::Node
{
public:
  SystemIntegration(const std::string & prompt)
  : Node("main"), prompt_(prompt)
  {
    // Subscribe to camera
    img_sub = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/camera/color/image_raw", 10,
      std::bind(&SystemIntegration::image_callback, this, std::placeholders::_1));

    // Create clients
    inference_srv = this->create_client<custom_interfaces_pkg::srv::Inference>(
      "service_send_to_model");

    arm_srv = this->create_client<custom_interfaces_pkg::srv::RobotCmd>(
      "arm_srv");

    // Wait for services
    while (!inference_srv->wait_for_service(1s)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for service_send_to_model...");
    }
    while (!arm_srv->wait_for_service(1s)) {
      RCLCPP_INFO(this->get_logger(), "Waiting for robot_cmd_service...");
    }
  }

  void run()
  {
    while(true){

    sensor_msgs::msg::Image::SharedPtr img;
    {
      std::lock_guard<std::mutex> lock(image_mutex_);
      if (!latest_image_) {
        RCLCPP_ERROR(this->get_logger(), "No image received yet.");
        return;
      }
      img = latest_image_;
    }

    // --- Call the AI service ---
    auto ai_req = std::make_shared<custom_interfaces_pkg::srv::Inference::Request>();
    ai_req->prompt = prompt_;
    ai_req->image = *img;

    auto ai_future = inference_srv->async_send_request(ai_req);

    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), ai_future)
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to call service_send_to_model");
      return;
    }

    auto ai_result = ai_future.get();
    RCLCPP_INFO(this->get_logger(), "Got response from AI service");

    // --- Call the Robot service ---
    auto robot_req = std::make_shared<custom_interfaces_pkg::srv::RobotCmd::Request>();
    robot_req->delta_position = ai_result->delta_position;
    robot_req->delta_orientation = ai_result->delta_orientation;
    robot_req->delta_gripper = ai_result->delta_gripper;

    auto robot_future = arm_srv->async_send_request(robot_req);

    if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), robot_future)
        != rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to call robot_cmd_service");
      return;
    }

    auto robot_result = robot_future.get();
    if (robot_result->success) {
      RCLCPP_INFO(this->get_logger(), "Robot executed command successfully!");
    } else {
      RCLCPP_WARN(this->get_logger(), "Robot failed to execute command.");
    }
  }
  }

private:
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    latest_image_ = msg;
  }

  std::string prompt_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
  rclcpp::Client<custom_interfaces_pkg::srv::Inference>::SharedPtr inference_srv;
  rclcpp::Client<custom_interfaces_pkg::srv::RobotCmd>::SharedPtr arm_srv;

  sensor_msgs::msg::Image::SharedPtr latest_image_;
  std::mutex image_mutex_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  if (argc < 2) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Usage: run_programme_client <prompt>");
    return 1;
  }

  std::string prompt = argv[1];
  auto node = std::make_shared<SystemIntegration>(prompt);

  // Give some time to collect an image
  RCLCPP_INFO(node->get_logger(), "Waiting for image...");
  rclcpp::Rate rate(10);
  while (rclcpp::ok() && !node->count_subscribers("/camera/camera/color/image_raw")) {
    rclcpp::spin_some(node);
    rate.sleep();
  }

  // Spin a bit until we actually receive an image
  for (int i = 0; i < 5 && rclcpp::ok(); ++i) {
    rclcpp::spin_some(node);
    rate.sleep();
  }

  node->run();

  rclcpp::shutdown();
  return 0;
}
