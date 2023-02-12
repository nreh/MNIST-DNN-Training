// Logging initialization and configuration

#include <iostream>
#include <glog/logging.h>
#include <termcolor/termcolor.hpp>

#define WITH_CUSTOM_PREFIX

namespace logging {

    // https://github.com/google/glog#custom-log-prefix-format
    void CustomPrefix(std::ostream& s, const google::LogMessageInfo& l, void*) {
        // custom colors based on severity
        if ((string)l.severity == "INFO") {
            cout << termcolor::green;
        } else if ((string)l.severity == "WARNING") {
            cout << termcolor::yellow;
        } else if ((string)l.severity == "ERROR") {
            cout << termcolor::red;
        } else if ((string)l.severity == "FATAL") {
            cout << termcolor::red << termcolor::bold;
        }

        cout << l.filename << ":" << l.line_number << " [" << l.severity << "]: ";

        cout << termcolor::reset;
    }

    void CustomFailureFunction() {
        cout << "...";
        return;
    }

    /**
     * @brief Initialize Google logging (glog)
     *
     * @param argv
     */
    void initialize(char* argv[]) {
        // Initialize Google’s logging library.
        google::InitGoogleLogging(argv[0], &CustomPrefix);

        // prevent exiting on ERROR and FATAL log entries
        google::InstallFailureFunction(&CustomFailureFunction);

        // so that INFO and other logging messages are printed
        FLAGS_logtostdout = 1;
    }
}