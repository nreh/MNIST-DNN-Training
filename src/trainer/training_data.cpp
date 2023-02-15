#include <string>
#include <fstream>
#include <sstream> // for parsing comma deliminated string

#include "../logging.h"

using namespace std;

/**
 * @brief Contains paths to training data and setter functions for quick validations. Also provides functions for reading
 *        data from files in batches
 */
class TrainingData {
public:

    // File readers

    ifstream* training_data_file = NULL;
    ifstream* training_labels_file = NULL;
    ifstream* test_data_file = NULL;
    ifstream* test_labels_file = NULL;

    /**
     * @brief Path to CSV file containing training inputs
     */
    string training_input_path = "";

    /**
     * @brief Path to CSV file containing training labels
     */
    string training_labels_path = "";

    /**
     * @brief Path to CSV file containing testing input
     */
    string test_input_path = "";

    /**
     * @brief Path to CSV file containing training labels
     */
    string test_labels_path = "";

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
            training_input_path = path;
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
            training_labels_path = path;
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
            test_input_path = path;
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
            test_labels_path = path;
        }
    }

    /**
     * @brief Split string by comma and parse floats into destination array
     *
     * @param record Training record to parse
     * @param destination Destination array to write values into - should already by initialized
     */
    void parse_record(string record, float* destination) {
        stringstream stream(record);

        int i = 0;
        while (stream.good()) {
            string temp;
            getline(stream, temp, ',');
            destination[i] = stof(temp);
            i += 1;
        }
    }

    /**
     * @brief Throws error if specified file stream is NULL. We have a separate function for this incase we want to add
     *        additional logic and to keep wording between errors the same.
     */
    void verify_file_open(ifstream* filestream, string type) {
        if (filestream == NULL) {
            throw invalid_function_call(type + " file has not been opened for reading");
        }
    }

    /**
     * @brief Get the next training batch from training data file
     *
     * @param data 2-D array that training batch will be written to. Should already be initialized.
     * @param batch_size How many training items to read from file
     *
     * @return Array containing training batch obtained from file
     */
    void get_next_training_data_batch(float** data, int batch_size) {
        verify_file_open(training_data_file, "Training data");

        for (int x = 0; x < batch_size; x++) {
            string record;
            getline(*training_data_file, record);
            parse_record(record, data[x]);
        }
    }

    /**
     * @brief Get the next training labels batch from training labels data file
     *
     * @param data 2-D array that training batch will be written to. Should already be initialized.
     * @param batch_size How many training items to read from file
     *
     * @return Array containing training batch obtained from file
     */
    void get_next_training_labels_batch(float** data, int batch_size) {
        verify_file_open(training_labels_file, "Training labels");

        for (int x = 0; x < batch_size; x++) {
            string record;
            getline(*training_labels_file, record);
            parse_record(record, data[x]);
        }
    }

    /**
     * @brief Get the next testing batch from training data file
     *
     * @param data 2-D array that training batch will be written to. Should already be initialized.
     * @param batch_size How many training items to read from file
     *
     * @return Array containing training batch obtained from file
     */
    void get_next_testing_data_batch(float** data, int batch_size) {
        verify_file_open(test_data_file, "Testing data");

        for (int x = 0; x < batch_size; x++) {
            string record;
            getline(*test_data_file, record);
            parse_record(record, data[x]);
        }
    }

    /**
     * @brief Get the next testing labels batch from training labels data file
     *
     * @param data 2-D array that training batch will be written to. Should already be initialized.
     * @param batch_size How many training items to read from file
     *
     * @return Array containing training batch obtained from file
     */
    void get_next_testing_labels_batch(float** data, int batch_size) {
        verify_file_open(test_data_file, "Testing labels");

        for (int x = 0; x < batch_size; x++) {
            string record;
            getline(*test_labels_file, record);
            parse_record(record, data[x]);
        }
    }

    ~TrainingData() {
        delete training_data_file, training_labels_file, test_data_file, test_labels_file;
    }

};