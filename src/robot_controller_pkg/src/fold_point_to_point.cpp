#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces_pkg/srv/fold_point_to_point.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <netinet/in.h>


std::string moveL(std::vector<double> pos, std::vector<double> ori, double speed, double acc, double blend)
{
    std::stringstream command;
    command << "movel(p[" << pos[0] << "," << pos[1] << "," << pos[2] << "," << ori[0] << "," << ori[1] << "," << ori[2] << "], a = "  << acc << ", v = " << speed << ", r = " << blend << ")\n";
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
    std::shared_ptr<custom_interfaces_pkg::srv::FoldPointToPoint::Response> send_command(const std::stringstream &command, int sock = 0, bool print_response = false)
    {
        std::shared_ptr<custom_interfaces_pkg::srv::FoldPointToPoint::Response> response = std::make_shared<custom_interfaces_pkg::srv::FoldPointToPoint::Response>();

        // Socket initialize
        try
        {
            struct sockaddr_in serv_addr;

            
            if ((sock = socket(AF_INET, SOCK_STREAM,0)) < 0){
                RCLCPP_INFO(this->get_logger(), "Erorr with socket (0)");
                response->succes = false;
                return response;
            }
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(PORT);

            if (inet_pton(AF_INET,IP.c_str(),&serv_addr.sin_addr) <= 0) {
                RCLCPP_INFO(this->get_logger(), "Erorr with socket (1)");
                response->succes = false;
                return response;
            }

            if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
                RCLCPP_INFO(this->get_logger(), "Erorr with socket (2)");
                response->succes = false;
                return response;
            }

            // Socket send command
            int result = send(sock, command.str().c_str(), command.str().size(), 0);
            char buffer[1024] = {0};

            // Wait for robot to process command and send response
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));

            // Socket receive response
            int valread = recv(sock, buffer, 1024, 0);
            if (print_response){
                RCLCPP_INFO(this->get_logger(), "Message from robot: \n%s", buffer);
            }
            

            close(sock);

            if (result == -1){
                response->succes = false;
                return response;
            }

            response->succes = true;
        }
        catch(const std::exception& e)
        {
            RCLCPP_INFO(this->get_logger(), "%s", e.what());
            response->succes = false;
        }

        return response;
    }


    int open_socket(int sock, int port)
    {
        struct sockaddr_in serv_addr;

        if ((sock = socket(AF_INET, SOCK_STREAM,0)) < 0){
            RCLCPP_INFO(this->get_logger(), "Erorr with socket (0)");
            return -1;
        }
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(PORT);

        if (inet_pton(AF_INET,IP.c_str(),&serv_addr.sin_addr) <= 0) {
            RCLCPP_INFO(this->get_logger(), "Erorr with socket (1)");
            return -1;
        }

        if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0){
            RCLCPP_INFO(this->get_logger(), "Erorr with socket (2)");
            return -1;
        }

        return sock;
    }

    int create_listening_socket(int port)
    {
        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) {
            RCLCPP_INFO(this->get_logger(), "Error creating server socket");
            return -1;
        }

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr;
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(port);

        if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            RCLCPP_INFO(this->get_logger(), "Error binding server socket");
            close(server_fd);
            return -1;
        }

        if (listen(server_fd, 1) < 0) {
            RCLCPP_INFO(this->get_logger(), "Error listening on server socket");
            close(server_fd);
            return -1;
        }

        return server_fd;
    }

    int accept_connection(int listen_sock, int timeout_ms = 5000)
    {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(listen_sock, &set);

        struct timeval tv;
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        int rv = select(listen_sock + 1, &set, NULL, NULL, &tv);
        if (rv <= 0) {
            // timeout or error
            return -1;
        }

        int client = accept(listen_sock, NULL, NULL);
        return client;
    }

    std::string read_from_socket(int sock)
    {
        std::string out;
        char buffer[1024];
        ssize_t n;
        // read until socket closes or no more data
        while ((n = recv(sock, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[n] = '\0';
            out += buffer;
            // small break if single message expected; otherwise continue reading
            if (n < (ssize_t)sizeof(buffer) - 1) break;
        }
        return out;
    }

    void close_socket(int sock)
    {
        if (sock > 0) close(sock);
    }

    std::vector<double> clean_get_actual_tcp_pose(std::string msg)
    {
        msg.replace(msg.find("p["), 2, ""); // remove leading 'p['
        msg.replace(msg.find("]"), 1, "");  // remove trailing ']'
        std::istringstream ss(msg);
        std::string token;
        std::vector<double> current_position;
        
        while (std::getline(ss, token, ',')) {
            try {
                current_position.push_back(std::stod(token));
            } catch (const std::invalid_argument& e) {
                RCLCPP_INFO(this->get_logger(), "Invalid number in position data: %s", token.c_str());
            }
        }
        if (current_position.size() != 6) {
            RCLCPP_INFO(this->get_logger(), "Unexpected number of position values: %zu", current_position.size());
        }

        return current_position;
    }


    void fold_point_to_point_srv_callback(
        const std::shared_ptr<custom_interfaces_pkg::srv::FoldPointToPoint::Request> request,
        std::shared_ptr<custom_interfaces_pkg::srv::FoldPointToPoint::Response> response)
    {
        std::vector<double> from_point = {request->from_point[0],request->from_point[1], request->from_point[2]};
        std::vector<double> to_point = {request->to_point[0],request->to_point[1], request->to_point[2]};
        std::vector<double> midpoint = {(from_point[0] + to_point[0])/2, (from_point[1] + to_point[1])/2, std::max(from_point[2], to_point[2]) + midpoint_extra_height};

        RCLCPP_INFO(this->get_logger(), "Service request received");

        // 1) Start listening on the PC BEFORE asking the robot to connect back
        int listen_sock = create_listening_socket(50000);
        if (listen_sock < 0) {
            RCLCPP_INFO(this->get_logger(), "Failed to start listening socket on port 50000");
            return;
        }

        std::stringstream command;
        command << ""
                << moveL(home, fixed_orientation, 0.25, 1.2, 0) 
                << openGripper()
                << moveL(from_point, fixed_orientation, 0.25, 1.2, 0)
                << closeGripper()
                << moveL(midpoint, fixed_orientation, 0.25, 1.2, 0.10)
                << moveL(to_point, fixed_orientation, 0.25, 1.2, 0)
                << openGripper()
                << moveL(home, fixed_orientation, 0.25, 1.2, 0)

                << "socket_open(\"192.168.1.104\", 50000, socket_name=\"socket_10\")\n"
                << "socket_send_string(to_str(get_actual_tcp_pose()), socket_name=\"socket_10\")\n"
                << "socket_close(socket_name=\"socket_10\")\n"
                << "";

        // 2) Send URScript to robot (robot will try to connect to this PC)
        send_command(command, 0, true);

        // 3) Accept the incoming connection from robot (with timeout)
        int client = accept_connection(listen_sock, 5000);
        if (client < 0) {
            RCLCPP_INFO(this->get_logger(), "No incoming connection from robot (timeout or error)");
            close_socket(listen_sock);
            return;
        }

        // 4) Read message from robot
        std::string msg = read_from_socket(client);
        RCLCPP_INFO(this->get_logger(), "Message from socket: \n%s", msg.c_str());

        // 5) Clean up the message and convert to vector
        std::vector<double> current_position = clean_get_actual_tcp_pose(msg);
        RCLCPP_INFO(this->get_logger(), "Current robot position: x=%.3f, y=%.3f, z=%.3f, rx=%.3f, ry=%.3f, rz=%.3f",
                    current_position[0], current_position[1], current_position[2],
                    current_position[3], current_position[4], current_position[5]);
        

        close_socket(client);
        close_socket(listen_sock);
    }


    rclcpp::Service<custom_interfaces_pkg::srv::FoldPointToPoint>::SharedPtr fold_point_to_point;
    const std::string IP = "192.168.1.100";
    const int PORT = 30020;

    const double midpoint_extra_height = 0.3;
    
    std::vector<double> picture_orientation = {3.1415/2, 0.0, -1.992};
    std::vector<double> picture_position = {-0.473, -0.230, 0.530};
    std::vector<double> grip_orientation = {3.1415, 0, 0};
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FoldPointToPoint>());
    rclcpp::shutdown();
    return 0;
}

/*
Service call example:
ros2 service call /fold_point_to_point_srv custom_interfaces_pkg/srv/FoldPointToPoint "{from_point: [0.0, 0.0, 0.0], tp_point: [0.0, 0.0, 0.0]}"
*/