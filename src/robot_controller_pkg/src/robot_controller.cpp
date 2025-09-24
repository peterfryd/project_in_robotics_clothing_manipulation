#include "rclcpp/rclcpp.hpp"
#include "robot_controller_pkg/msg/robot_cmd.hpp"
#include "robot_controller_pkg/srv/robot_cmd.hpp"
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <modbus/modbus.h>
#include <stdexcept>
#include <vector>
#include <string>

// ------------------- Gripper class -------------------
class OnRobot2FG7
{
public:
    OnRobot2FG7(const std::string &ip, int port = 502)
    {
        ctx = modbus_new_tcp(ip.c_str(), port);
        if (!ctx)
            throw std::runtime_error("Unable to allocate Modbus context");

        if (modbus_connect(ctx) == -1)
        {
            modbus_free(ctx);
            throw std::runtime_error("Unable to connect to 2FG7");
        }
    }

    ~OnRobot2FG7()
    {
        modbus_close(ctx);
        modbus_free(ctx);
    }

    void setGripper(bool close)
    {
        uint16_t value = close ? 255 : 0;                 // 0=open, 255=close (check manual)
        int rc = modbus_write_register(ctx, 1000, value); // 1000 = "target position" register
        if (rc == -1)
            throw std::runtime_error("Failed to write to gripper");
    }

private:
    modbus_t *ctx;
};

// ------------------- ROS2 Node -------------------
class RobotController : public rclcpp::Node
{
public:
    RobotController()
        : Node("robot_controller"),
          rtde_c("192.168.1.100"),
          rtde_r("192.168.1.100"),
          gripper("192.168.1.101") // IP of your 2FG7
    {
        robot_cmd_subscriber = this->create_subscription<robot_controller_pkg::msg::RobotCmd>(
            "robot_cmd", 10,
            std::bind(&RobotController::RobotCmdCallback, this, std::placeholders::_1));

        robot_cmd_service = this->create_service<robot_controller_pkg::srv::RobotCmd>(
            "robot_cmd_service",
            std::bind(&RobotController::handle_robot_cmd, this,
                      std::placeholders::_1, std::placeholders::_2));
    }

private:
    void RobotCmdCallback(const robot_controller_pkg::msg::RobotCmd::SharedPtr msg)
    {
        std::vector<double> pose = rtde_r.getActualTCPPose();

        std::vector<double> new_pose = {
            pose[0] + msg->delta_position[0],
            pose[1] + msg->delta_position[1],
            pose[2] + msg->delta_position[2],
            pose[3] + msg->delta_orientation[0],
            pose[4] + msg->delta_orientation[1],
            pose[5] + msg->delta_orientation[2]};

        // Move robot
        rtde_c.moveL(new_pose, 0.25, 0.25);

        // Control gripper via Modbus
        // RCLCPP_INFO(this->get_logger(), "Setting gripper to %s", msg->delta_gripper > 0.5 ? "CLOSE" : "OPEN");
        // gripper.setGripper(msg->delta_gripper > 0.5);
    }

    void handle_robot_cmd(
        const std::shared_ptr<robot_controller_pkg::srv::RobotCmd::Request> request,
        std::shared_ptr<robot_controller_pkg::srv::RobotCmd::Response> response)
    {
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

            // Move robot
            rtde_c.moveL(new_pose, 0.25, 0.25);

            // Example for gripper (uncomment if you have gripper class)
            // gripper.setGripper(request->delta_gripper > 0.5);

            response->success = true;
            response->message = "Command executed";
        }
        catch (const std::exception &e)
        {
            response->success = false;
            response->message = e.what();
        }
    }

    rclcpp::Subscription<robot_controller_pkg::msg::RobotCmd>::SharedPtr robot_cmd_subscriber;
    rclcpp::Service<robot_controller_pkg::srv::RobotCmd>::SharedPtr robot_cmd_service;
    ur_rtde::RTDEControlInterface rtde_c;
    ur_rtde::RTDEReceiveInterface rtde_r;
    OnRobot2FG7 gripper;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotController>());
    rclcpp::shutdown();
    return 0;
}
