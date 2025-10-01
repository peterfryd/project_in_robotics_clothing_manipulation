#include "rclcpp/rclcpp.hpp"
#include "custom_interfaces_pkg/srv/robot_cmd.hpp"
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <modbus/modbus.h>
#include <stdexcept>
#include <vector>
#include <string>

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

int main(int argc, char *argv[])
{
    return 0;
}
