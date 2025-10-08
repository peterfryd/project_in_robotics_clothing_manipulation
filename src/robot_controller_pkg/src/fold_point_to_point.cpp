#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces_pkg/srv/fold_point_to_point.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>


std::string moveL(std::vector<double> pos, std::vector<double> ori, double speed, double acc, double blend)
{
    std::stringstream command;
    command << "movel(p[" << pos[0] << "," << pos[1] << "," << pos[2] << "," << ori[0] << "," << ori[1] << "," << ori[2] << "], a = "  << acc << ", v = " << speed << " )\n";
    return command.str();
}

std::string setDigitalOut(int index, bool value)
{
    std::stringstream command;
    command << "set_standard_digital_out(" << index << "," << "True" << ")\n";
    return command.str();
}

std::string setGripperWidth(double width)
{
    std::stringstream command;
    command << "twofg_release_ext(" << width << ", 50, 0)\n";
    return command.str();
}

std::string openGripper()
{
    return setGripperWidth(71);
}

std::string closeGripper()
{
    return setGripperWidth(33);
}



class FoldPointToPoint : public rclcpp::Node
{
public:
    FoldPointToPoint()
        : Node("fold_point_to_point")
    {
        fold_point_to_point = this->create_service<custom_interfaces_pkg::srv::FoldPointToPoint>(
            "/fold_point_to_point_srv",
            std::bind(&FoldPointToPoint::fold_point_to_point_srv_callback, this,
                      std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "Node started");
    }

private:
    void fold_point_to_point_srv_callback(
        const std::shared_ptr<custom_interfaces_pkg::srv::FoldPointToPoint::Request> request,
        std::shared_ptr<custom_interfaces_pkg::srv::FoldPointToPoint::Response> response)
    {
        std::vector<double> fixed_orientation = {3.1415, 0, 0};
        std::vector<double> from_point = {request->from_point[0],request->from_point[1], request->from_point[2]};
        std::vector<double> to_point = {request->to_point[0],request->to_point[1], request->to_point[2]};

        
            
        RCLCPP_INFO(this->get_logger(), "Service request received");

        std::stringstream command;

        // Home
        std::vector<double> home = {-0.4, -0.0, 0.3};
        // std::vector<double> pointA = {-0.120, -0.350, 0.05};
        // std::vector<double> pointB = {-0.680, -0.210, 0.05};

        // command = command + setDigitalOut(0, true);
        command << moveL(home, fixed_orientation, 0.25, 1.2, 0) 
                << openGripper()
                << moveL(from_point, fixed_orientation, .25, 1.2, 0)
                << closeGripper()
                << moveL(to_point, fixed_orientation, 0.25, 1.2, 0)
                << openGripper()
                << moveL(home, fixed_orientation, 0.25, 1.2, 0);




        
        // Socket initialize and send
        try
        {
            int sock = 0;
            struct sockaddr_in serv_addr;

            
            if ((sock = socket(AF_INET, SOCK_STREAM,0)) < 0){
                RCLCPP_INFO(this->get_logger(), "Erorr with socket (0)");
                response->succes = false;
                return;
            }
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(PORT);

            if (inet_pton(AF_INET,IP.c_str(),&serv_addr.sin_addr) <= 0) {
                RCLCPP_INFO(this->get_logger(), "Erorr with socket (1)");
                response->succes = false;
                return;
            }

            if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
                RCLCPP_INFO(this->get_logger(), "Erorr with socket (2)");
                response->succes = false;
                return;
            }

            int result = send(sock, command.str().c_str(), command.str().size(), 0);
            close(sock);

            if (result == -1){
                response->succes = false;
                return;
            }

            response->succes = true;
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO(this->get_logger(), e.what());
            response->succes = false;
        }
    }

    rclcpp::Service<custom_interfaces_pkg::srv::FoldPointToPoint>::SharedPtr fold_point_to_point;
    const std::string IP = "192.168.1.100";
    const int PORT = 30020;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FoldPointToPoint>());
    rclcpp::shutdown();
    return 0;
}