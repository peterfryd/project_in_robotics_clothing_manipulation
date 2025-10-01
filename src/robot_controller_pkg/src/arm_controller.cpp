#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces_pkg/srv/robot_cmd.hpp"
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <modbus/modbus.h>
#include <stdexcept>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

// Convert rotation vector (axis-angle) to rotation matrix
Eigen::Matrix3d rotVecToRotMat(const Eigen::Vector3d& rvec) {
    double theta = rvec.norm();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (theta < 1e-12) {
        // Zero rotation -> identity
        return R;
    }

    Eigen::Vector3d axis = rvec / theta;

    // Skew-symmetric matrix of axis
    Eigen::Matrix3d K;
    K <<       0, -axis.z(),  axis.y(),
          axis.z(),        0, -axis.x(),
         -axis.y(),  axis.x(),        0;

    R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;

    return R;
}



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
            std::vector<double> gripper_delta;
            for (double& del: request->delta_position){
                gripper_delta.push_back(del);
            }
            
            Eigen::Vector3d position; // meters
            Eigen::Vector3d rotVec; // radians

            position << pose[0], pose[1], pose[2];
            rotVec << pose[3], pose[4], pose[5];

            Eigen::Matrix3d R = rotVecToRotMat(rotVec);
            
            Eigen::Vector3d delta;
            delta << gripper_delta[0], gripper_delta[1], gripper_delta[2];

            // Express delta in base frame
            Eigen::Vector3d delta_in_base = R * delta;

            // New TCP position in base frame
            Eigen::Vector3d new_position = position + delta_in_base;

            std::vector<double> new_position_vec;
            for (double& del: new_position){
                new_position_vec.push_back(del);
            }
            
            rtde_c.moveL(new_position_vec, 0.25, 0.25);
            
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