// Logging initialization and configuration

#include <iostream>
#include <termcolor/termcolor.hpp>
#include <spdlog/spdlog.h>

namespace logging {

    /**
     * @brief Initialize logging
     *
     * @param argv
     */
    void initialize(char* argv[]) {
        spdlog::set_level(spdlog::level::debug); // Set global log level to debug

        // change log pattern
        spdlog::set_pattern("[%^%l%$]: %v");
    }
}