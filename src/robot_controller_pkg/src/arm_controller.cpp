#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces_pkg/srv/robot_cmd.hpp"
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <modbus/modbus.h>
#include <stdexcept>
#include <vector>
#include <string>

class ArmController : public rclcpp::Node
{
public:
    ArmController()
        : Node("arm_controller"), rtde_c("192.168.1.100"), rtde_r("192.168.1.100"){

        arm_srv = this->create_service<custom_interfaces_pkg::srv::RobotCmd>(
            "arm_srv",
            std::bind(&ArmController::handle_robot_cmd, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

private:

    void handle_robot_cmd(
        const std::shared_ptr<custom_interfaces_pkg::srv::RobotCmd::Request> request,
        std::shared_ptr<custom_interfaces_pkg::srv::RobotCmd::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Service request received");
        try
        {
            std::vector<double> pose = rtde_r.getActualTCPPose();

            std::vector<double> new_pose = {
                pose[0] + request->delta_position[0],
                pose[1] + request->delta_position[1],
                pose[2] + request->delta_position[2],
                pose[3] + request->delta_orientation[0],
                pose[4] + request->delta_orientation[1],
                pose[5] + request->delta_orientation[2]};

            std::vector<double> delta_pose = {
                pose[0] + request->delta_position[0],
                pose[1] + request->delta_position[1],
                pose[2] + request->delta_position[2],
                pose[3] + request->delta_orientation[0],
                pose[4] + request->delta_orientation[1],
                pose[5] + request->delta_orientation[2]};

            // Move robot
            movel(pose_tans(get_actual_tcp_pose(),delta_pose), 0.25, 0.25);    # Transform to TCP-space
            // rtde_c.moveL(new_pose, 0.25, 0.25);
            RCLCPP_INFO(this->get_logger(), "Robot moved to new position");

            response->success = true;
        }
        catch (const std::exception &e)
        {
            RCLCPP_INFO(this->get_logger(), "Error executing command: %s", e.what());
            response->success = false;
        }
        RCLCPP_INFO(this->get_logger(), "Service response sent");
    }

    rclcpp::Service<custom_interfaces_pkg::srv::RobotCmd>::SharedPtr arm_srv;
    ur_rtde::RTDEControlInterface rtde_c;
    ur_rtde::RTDEReceiveInterface rtde_r;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmController>());
    rclcpp::shutdown();
    return 0;
}