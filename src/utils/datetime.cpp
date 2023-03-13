#pragma once

// https://stackoverflow.com/a/16358111/5614052

#include <ctime>
#include <iomanip>
#include <sstream>

std::string get_current_datetime() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%B-%Y_%H:%M:%S");
    auto str = oss.str();

    return str;
}