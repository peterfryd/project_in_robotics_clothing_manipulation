#include <memory>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <typeinfo>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/empty.hpp"

#include "custom_interfaces_pkg/srv/get_pick_and_place_point.hpp"
#include "custom_interfaces_pkg/srv/image_to_base.hpp"
#include "custom_interfaces_pkg/srv/fold_point_to_point.hpp"
#include "custom_interfaces_pkg/srv/get_landmarks.hpp"

using namespace std::chrono_literals;

class SystemIntegration : public rclcpp::Node
{
public:
    SystemIntegration(const std::string &prompt)
        : Node("main"), prompt_(prompt)
    {
        // Create service clients
        get_pick_and_place_srv = this->create_client<custom_interfaces_pkg::srv::GetPickAndPlacePoint>(
            "/get_pick_and_place_point_srv"
        );

        image_to_base_srv = this->create_client<custom_interfaces_pkg::srv::ImageToBase>(
            "/image_to_base_srv"
        );

        fold_point_to_point_srv = this->create_client<custom_interfaces_pkg::srv::FoldPointToPoint>(
            "/fold_point_to_point_srv"
        );

        get_landmarks_srv = this->create_client<custom_interfaces_pkg::srv::GetLandmarks>(
            "/get_landmarks_srv"
        );

        fold_point_to_point_home_srv = this->create_client<std_srvs::srv::Empty>(
            "/fold_point_to_point_home_srv"
        );

        update_background_image_srv = this->create_client<std_srvs::srv::Empty>(
            "/update_background_image_srv"
        );

        // Wait for services
        while (!get_pick_and_place_srv->wait_for_service(1s))
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for get_pick_and_place_point_srv...");
        }
        while (!image_to_base_srv->wait_for_service(1s))
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for image_to_base_srv...");
        }
        while (!fold_point_to_point_srv->wait_for_service(1s))
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for fold_point_to_point_srv...");
        }
        while (!update_background_image_srv->wait_for_service(1s))
        {
            RCLCPP_INFO(this->get_logger(), "Waiting for update_background_image_srv...");
        }
    }

    int run()
    {

        int step = std::stoi(prompt_);

        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Running!");


        // Move to home position
        auto fold_point_to_point_home_req = std::make_shared<std_srvs::srv::Empty::Request>();
        auto fold_point_to_point_home_future = fold_point_to_point_home_srv->async_send_request(fold_point_to_point_home_req);

        if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), fold_point_to_point_home_future)
            != rclcpp::FutureReturnCode::SUCCESS)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to call /fold_point_to_point_home_srv");
            return 11;
        }

        //std::this_thread::sleep_for(std::chrono::milliseconds(5000));
        RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Moved to home!");


        if(step == 1){
            // Get landmarks
            auto get_landmarks_req = std::make_shared<custom_interfaces_pkg::srv::GetLandmarks::Request>();
            auto get_landmarks_future = get_landmarks_srv->async_send_request(get_landmarks_req);

            if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), get_landmarks_future)
                != rclcpp::FutureReturnCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to call /get_landmarks_srv");
                return 10;
            }

            auto get_landmarks_result = get_landmarks_future.get();
            RCLCPP_INFO(this->get_logger(), "Got response from /get_landmarks_srv service");
            std::string landmarks_type = typeid(get_landmarks_result).name();
            RCLCPP_INFO(this->get_logger(), "get_landmarks_result: %s", landmarks_type.c_str());
            RCLCPP_INFO(this->get_logger(), "Landmark 0: x = %f, y = %f", get_landmarks_result->landmarks[0].x, get_landmarks_result->landmarks[0].y);
        }

        if (prompt_ != ""){
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Folding step %i", step);
            
            auto get_pick_and_place_req = std::make_shared<custom_interfaces_pkg::srv::GetPickAndPlacePoint::Request>();
            get_pick_and_place_req->step_number = step;

            auto get_pick_and_place_future = get_pick_and_place_srv->async_send_request(get_pick_and_place_req);

            if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), get_pick_and_place_future)
                != rclcpp::FutureReturnCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to call /get_pick_and_place_point_srv");
                return 2;
            }

            auto get_pick_and_place_point_result = get_pick_and_place_future.get();

            std::array<int, 2> pick_point = get_pick_and_place_point_result->image_pick_point;
            std::array<int, 2> place_point = get_pick_and_place_point_result->image_place_point;

            RCLCPP_INFO(this->get_logger(), "Got response from /get_pick_and_place_srv service: pick_point = {%i, %i} place_point = {%i, %i}", pick_point[0], pick_point[1], place_point[0], place_point[1]);

            // Check valid return
            if (pick_point[0] == -1 || pick_point[1] == -1 || place_point[0] == -1 || place_point[1] == -1){
                RCLCPP_ERROR(this->get_logger(), "Not valid returns!");
                return 3;
            }

            
            // Convert pick point
            auto image_to_base_req = std::make_shared<custom_interfaces_pkg::srv::ImageToBase::Request>();
            image_to_base_req->imageframe_coordinates = pick_point;

            RCLCPP_INFO(this->get_logger(), "1");

            auto image_to_base_future_pick = image_to_base_srv->async_send_request(image_to_base_req);
            RCLCPP_INFO(this->get_logger(), "2");
            if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), image_to_base_future_pick)
                != rclcpp::FutureReturnCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to call /get_pick_and_place_point_srv");
                return 2;
            }
            RCLCPP_INFO(this->get_logger(), "3");

            auto image_to_base_result_pick = image_to_base_future_pick.get();
            std::array<double, 3>  pick_point_baseframe = image_to_base_result_pick->baseframe_coordinates;
            pick_point_baseframe[2] = -0.0125;

            RCLCPP_INFO(this->get_logger(), "Got response from /image_to_base service: pick_baseframe = {%f, %f, %f}", pick_point_baseframe[0], pick_point_baseframe[1], pick_point_baseframe[2]);


            // Convert place point
            auto image_to_base_req_place = std::make_shared<custom_interfaces_pkg::srv::ImageToBase::Request>();
            image_to_base_req_place->imageframe_coordinates = place_point;

            auto image_to_base_future_place = image_to_base_srv->async_send_request(image_to_base_req_place);

            if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), image_to_base_future_place)
                != rclcpp::FutureReturnCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to call /get_pick_and_place_point_srv");
                return 3;
            }

            auto image_to_base_result_place = image_to_base_future_place.get();
            std::array<double, 3>  place_point_baseframe = image_to_base_result_place->baseframe_coordinates;
            place_point_baseframe[2] = -0.0;

            RCLCPP_INFO(this->get_logger(), "Got response from /image_to_base service: place_baseframe = {%f, %f, %f}", place_point_baseframe[0], place_point_baseframe[1], place_point_baseframe[2]);


            // Fold point to point
            auto fold_point_to_point_req = std::make_shared<custom_interfaces_pkg::srv::FoldPointToPoint::Request>();
            fold_point_to_point_req->from_point = pick_point_baseframe;
            fold_point_to_point_req->to_point = place_point_baseframe;

            auto fold_point_to_point_future = fold_point_to_point_srv->async_send_request(fold_point_to_point_req);

            if (rclcpp::spin_until_future_complete(this->get_node_base_interface(), fold_point_to_point_future)
                != rclcpp::FutureReturnCode::SUCCESS)
            {
                RCLCPP_ERROR(this->get_logger(), "Failed to call /fold_point_to_point_srv");
                return 4;
            }
            
            RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Moving through points!");
            // std::this_thread::sleep_for(std::chrono::milliseconds(10000));

            auto fold_point_to_point_result = fold_point_to_point_future.get();
            bool success = fold_point_to_point_result->succes;

            RCLCPP_INFO(this->get_logger(), "Got response from /fold_point_to_point_srv service: succes = %i", success);

            if (!success){
                RCLCPP_ERROR(this->get_logger(), "/fold_point_to_point_srv return success=false");
                return 5;
            }
        }

        return 0;
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        latest_image_ = msg;
    }

    std::string prompt_;

    rclcpp::Client<custom_interfaces_pkg::srv::GetPickAndPlacePoint>::SharedPtr get_pick_and_place_srv;
    rclcpp::Client<custom_interfaces_pkg::srv::ImageToBase>::SharedPtr image_to_base_srv;
    rclcpp::Client<custom_interfaces_pkg::srv::FoldPointToPoint>::SharedPtr fold_point_to_point_srv;
    rclcpp::Client<custom_interfaces_pkg::srv::GetLandmarks>::SharedPtr get_landmarks_srv;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr fold_point_to_point_home_srv;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr update_background_image_srv;
    
    // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub;
    // rclcpp::Client<custom_interfaces_pkg::srv::Inference>::SharedPtr inference_srv;
    // rclcpp::Client<custom_interfaces_pkg::srv::RobotCmd>::SharedPtr arm_srv;

    sensor_msgs::msg::Image::SharedPtr latest_image_;
    std::mutex image_mutex_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    if (argc > 2){
        RCLCPP_ERROR(rclcpp::get_logger("rclcpp"), "Too many arguments. Max 1 allowed!");
        return 1;
    }

    std::string step_number = "";
    if (argc > 1){
        step_number = argv[1];
    }

    auto node = std::make_shared<SystemIntegration>(step_number);

    // rclcpp::spin_some(node);
    // rate.sleep();

    node->run();

    rclcpp::shutdown();
    return 0;
}
