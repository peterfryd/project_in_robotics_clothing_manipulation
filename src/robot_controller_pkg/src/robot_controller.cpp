#include "rclcpp/rclcpp.hpp"
#include "robot_controller_pkg/msg/robot_cmd.hpp"
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>

class RobotController : public rclcpp::Node
{
public:
  RobotController() : Node("robot_controller"), rtde_c("192.168.1.100"), rtde_r("192.168.1.100")
  {
    robot_cmd_subscriber = this->create_subscription<robot_controller_pkg::msg::RobotCmd>(
        "robot_cmd", 10,
        std::bind(&RobotController::RobotCmdCallback, this, std::placeholders::_1));
  }

private:
  void RobotCmdCallback(const robot_controller_pkg::msg::RobotCmd::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(),
                "Received pos(%.2f, %.2f, %.2f) orient(%.2f, %.2f, %.2f) gripper=%.2f",
                msg->delta_position[0], msg->delta_position[1], msg->delta_position[2],
                msg->delta_orientation[0], msg->delta_orientation[1], msg->delta_orientation[2],
                msg->delta_gripper);

    std::vector<double> pose = rtde_r.getActualTCPPose();

    std::vector<double> new_pose = {
      pose[0] + msg->delta_position[0],
      pose[1] + msg->delta_position[1],
      pose[2] + msg->delta_position[2],
      pose[3] + msg->delta_orientation[0],
      pose[4] + msg->delta_orientation[1],
      pose[5] + msg->delta_orientation[2]
    };

    rtde_c.moveL(new_pose, 0.25, 0.25);
  }

  rclcpp::Subscription<robot_controller_pkg::msg::RobotCmd>::SharedPtr robot_cmd_subscriber;
  ur_rtde::RTDEControlInterface rtde_c;
  ur_rtde::RTDEReceiveInterface rtde_r;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RobotController>());
  rclcpp::shutdown();
  return 0;
}
