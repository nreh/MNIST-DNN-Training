// Logging initialization and configuration

#include <iostream>
#include <termcolor/termcolor.hpp>

#include "logging.h"

namespace logging {

    /**
     * @brief Initialize logging
     *
     * @param argv
     */
    void initialize(char* argv[]) {
        spdlog::set_level(spdlog::level::trace); // Set global log level to trace

        // change log pattern
        spdlog::set_pattern("src/%s:%# %^[%l]: %v%$");
    }
}