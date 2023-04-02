// Logging initialization and configuration

#include <iostream>
#include <termcolor/termcolor.hpp>

#include "logging.h"

namespace logging {

/**
 * @brief Initialize logging
 *
 * @param verbose
 */
void initialize(bool verbose) {
    if (verbose) {
        SPDLOG_INFO("Logging level set to VERBOSE");
        spdlog::set_level(spdlog::level::trace); // Set global log level to trace
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    // change log pattern
    spdlog::set_pattern("%-20s@ %-25!!%^[%l]: %v%$");
}
} // namespace logging