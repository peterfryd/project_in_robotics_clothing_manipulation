#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces_pkg/srv/robot_cmd.hpp"
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

// Convert rotation vector (axis-angle) to rotation matrix
Eigen::Matrix3d rotVecToRotMat(const Eigen::Vector3d &rvec)
{
    double theta = rvec.norm();

    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    if (theta < 1e-12)
    {
        // Zero rotation -> identity
        return R;
    }

    Eigen::Vector3d axis = rvec / theta;

    // Skew-symmetric matrix of axis
    Eigen::Matrix3d K;
    K << 0, -axis.z(), axis.y(),
        axis.z(), 0, -axis.x(),
        -axis.y(), axis.x(), 0;

    R = Eigen::Matrix3d::Identity() + std::sin(theta) * K + (1 - std::cos(theta)) * K * K;

    return R;
}

class ArmController : public rclcpp::Node
{
public:
    ArmController()
        : Node("arm_controller"), rtde_c("192.168.1.100"), rtde_r("192.168.1.100")
    {

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
            RCLCPP_INFO(this->get_logger(), "Start of try catch");
            // Get current TCP pose
            std::vector<double> pose = rtde_r.getActualTCPPose();
            Eigen::Vector3d position(pose[0], pose[1], pose[2]);
            Eigen::Vector3d rotVec(pose[3], pose[4], pose[5]);

            // Convert current rotation vector to rotation matrix
            Eigen::Matrix3d R_current = rotVecToRotMat(rotVec);

            // Translation delta in TCP frame
            Eigen::Vector3d delta_position(request->delta_position[0],
                                           request->delta_position[1],
                                           request->delta_position[2]);

            // Transform translation delta to base frame
            Eigen::Vector3d delta_in_base = R_current * delta_position;

            // New TCP position
            Eigen::Vector3d new_position = position + delta_in_base;

            // Rotation delta
            Eigen::Vector3d delta_orientation(request->delta_orientation[0],
                                              request->delta_orientation[1],
                                              request->delta_orientation[2]);

            // Convert delta rotation to matrix
            Eigen::Matrix3d R_delta = rotVecToRotMat(delta_orientation);

            // New rotation matrix (combine current and delta rotations)
            Eigen::Matrix3d R_new = R_current * R_delta;

            // Convert back to rotation vector
            Eigen::AngleAxisd angleAxis(R_new);
            Eigen::Vector3d new_rotVec = angleAxis.axis() * angleAxis.angle();

            // Build full 6-DOF TCP pose
            std::vector<double> new_pose = {
                new_position[0], new_position[1], new_position[2],
                new_rotVec[0], new_rotVec[1], new_rotVec[2]};

            // Debug print
            RCLCPP_INFO(this->get_logger(), "Moving to: [%f, %f, %f, %f, %f, %f]",
                        new_pose[0], new_pose[1], new_pose[2],
                        new_pose[3], new_pose[4], new_pose[5]);

            // Move the robot
            rtde_c.moveL(new_pose, 0.25, 0.25);
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