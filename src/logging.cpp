// Logging initialization and configuration

#include <iostream>
#include <glog/logging.h>

namespace logging {

    // https://github.com/google/glog#custom-log-prefix-format
    void CustomPrefix(std::ostream& s, const google::LogMessageInfo& l, void*) {
        s << l.filename << ":" << l.line_number << " [" << l.severity << "]:";
    }

    /**
     * @brief Initialize Google logging (glog)
     *
     * @param argv
     */
    void initialize(char* argv[]) {
        // Initialize Google’s logging library.
        google::InitGoogleLogging(argv[0], &CustomPrefix);

        // so that INFO and other logging messages are printed
        FLAGS_logtostderr = 1;
    }
}