#pragma once

#include <stdexcept>
#include <string>

/**
 * https://stackoverflow.com/a/17438895
 */

/**
 * @brief Thrown when an invalid function is invoked.
 */
class invalid_function_call : public std::exception {
  private:
    std::string message;

  public:
    explicit invalid_function_call(const std::string &message);
    const char *what() const noexcept override { return this->message.c_str(); }
};

invalid_function_call::invalid_function_call(const std::string &message) : message(message) {}