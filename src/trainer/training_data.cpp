#include <string>
#include <fstream>

#include "../logging.h"

using namespace std;

/**
 * @brief Contains paths to training data and setter methods for quick validations
 */
class TrainingData {
public:
    /**
     * @brief Path to CSV file containing training inputs
     */
    string training_input_file = "";

    /**
     * @brief Path to CSV file containing training labels
     */
    string training_labels_file = "";

    /**
     * @brief Path to CSV file containing testing input
     */
    string test_input_file = "";

    /**
     * @brief Path to CSV file containing training labels
     */
    string test_labels_file = "";

    /**
     * @brief Set the training input file and verify its existence
     *
     * @param path
     */
    void set_training_input_file(string path) {
        ifstream file_reader(path);

        if (file_reader.fail()) {
            SPDLOG_ERROR("Unable to open training input file '" + path + "'");
        } else {
            training_input_file = path;
        }
    }

    /**
     * @brief Set the training labels file and verify its existence
     *
     * @param path
     */
    void set_training_labels_file(string path) {
        ifstream file_reader(path);

        if (file_reader.fail()) {
            SPDLOG_ERROR("Unable to open training labels file '" + path + "'");
        } else {
            training_labels_file = path;
        }
    }

    /**
     * @brief Set the test input file and verify its existence
     *
     * @param path
     */
    void set_test_input_file(string path) {
        ifstream file_reader(path);

        if (file_reader.fail()) {
            SPDLOG_ERROR("Unable to open test input file '" + path + "'");
        } else {
            test_input_file = path;
        }
    }

    /**
     * @brief Set the test labels file and verify its existence
     *
     * @param path
     */
    void set_test_labels_file(string path) {
        ifstream file_reader(path);

        if (file_reader.fail()) {
            SPDLOG_ERROR("Unable to open test labels file '" + path + "'");
        } else {
            test_labels_file = path;
        }
    }
};