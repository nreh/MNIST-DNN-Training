#include <fstream>
#include <sstream> // for parsing comma deliminated string
#include <string>

#include "../logging.h"
#include "../utils/endian.cpp"

using namespace std;

/**
 * @brief Contains paths to training data and setter functions for quick validations. Also provides functions for reading
 *        data from files in batches
 */
class TrainingData {
  private:
  public:
    /**
     * @brief Number of items in single batch
     */
    int batch_size = 100;

    // File readers

    ifstream *training_data_file = NULL;
    ifstream *training_labels_file = NULL;
    ifstream *test_data_file = NULL;
    ifstream *test_labels_file = NULL;

    /**
     * @brief Path to binary file containing training inputs
     */
    string training_data_path = "";

    /**
     * @brief Path to binary file containing training labels
     */
    string training_labels_path = "";

    /**
     * @brief Path to binary file containing testing input
     */
    string test_data_path = "";

    /**
     * @brief Path to binary file containing training labels
     */
    string test_labels_path = "";

    /**
     * @brief Index of next record in next batch to be trained on
     */
    int current_record = 0;

    /**
     * @brief Number of rows in input image to network
     */
    int32_t input_rows = 0;

    /**
     * @brief Number of columns in input image to network
     */
    int32_t input_columns = 0;

    /**
     * @brief Number of total items in the training data set
     */
    int32_t training_data_items_count = 0;

    /**
     * @brief Number of total items in the test data set
     */
    int32_t test_data_items_count = 0;

    /**
     * @brief Index of current batch being trained on
     */
    int current_batch = 0;

    /**
     * @brief Total batches in training data set
     */
    int total_batch_count = 0;

    /**
     * @brief 2-D array containing training batch data. Because batch size doesn't change during training, we initialize
     *        this before hand so we don't need to reallocate memory every batch.
     */
    float **training_data_batch_buffer = NULL;

    /**
     * @brief 1-D array containing training batch labels. Because batch size doesn't change during training, we initialize
     *        this before hand so we don't need to reallocate memory every batch.
     */
    unsigned char *training_labels_batch_buffer = NULL;

    /**
     * @brief 2-D array containing test data.
     */
    float **test_data_buffer = NULL;

    /**
     * @brief 1-D array containing test data labels.
     */
    unsigned char *test_labels_buffer = NULL;

    /**
     * @brief Set the training input file and verify its existence
     *
     * @param path
     */
    void set_training_data_file(string path) {
        SPDLOG_INFO("Opening training data file '" + path + "' ...");

        delete training_data_file;

        training_data_path = path;
        training_data_file = new ifstream(path, ios::in | ios::binary);

        if (training_data_file->fail()) {
            delete training_data_file;
            training_data_file = NULL;
            throw invalid_argument("Unable to open training input file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            training_data_file->seekg(4); // skip 'magic number'
            training_data_items_count = file_read_big_endian_int32(*training_data_file);
            input_rows = file_read_big_endian_int32(*training_data_file);
            input_columns = file_read_big_endian_int32(*training_data_file);

            total_batch_count = (int)ceil(training_data_items_count / (float)batch_size);

            SPDLOG_DEBUG("count = " + to_string(training_data_items_count) + ", rows,cols = " + to_string(input_rows) + "," +
                         to_string(input_columns));

            // initialize training data bufferarray, first delete old one incase batch size or bytes-per-item changes
            if (training_data_batch_buffer != NULL) {
                for (int x = 0; x < batch_size; x++) {
                    delete[] training_data_batch_buffer[x];
                }
                delete[] training_data_batch_buffer;
            }

            const int values_per_input = input_rows * input_columns;

            training_data_batch_buffer = new float *[batch_size];
            for (int x = 0; x < batch_size; x++) {
                training_data_batch_buffer[x] = new float[values_per_input];
            }
        }
    }

    /**
     * @brief Set the training labels file and verify its existence
     *
     * @param path
     */
    void set_training_labels_file(string path) {
        SPDLOG_INFO("Opening training labels file '" + path + "' ...");

        delete training_labels_file;

        training_labels_path = path;
        training_labels_file = new ifstream(path, ios::in | ios::binary);

        if (training_labels_file->fail()) {
            delete training_labels_file;
            training_labels_file = NULL;
            throw invalid_argument("Unable to open training labels file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            training_labels_file->seekg(4); // skip 'magic number'
            training_data_items_count = file_read_big_endian_int32(*training_labels_file);

            SPDLOG_DEBUG("count = " + to_string(training_data_items_count));

            // initialize training labels bufferarray, first delete old one incase batch size changes
            if (training_labels_batch_buffer != NULL) {
                delete[] training_labels_batch_buffer;
            }

            training_labels_batch_buffer = new unsigned char[batch_size];
        }
    }

    /**
     * @brief Set the test input file and verify its existence
     *
     * @param path
     */
    void set_test_data_file(string path) {
        SPDLOG_INFO("Opening test data file '" + path + "' ...");

        delete test_data_file;

        test_data_path = path;
        test_data_file = new ifstream(path, ios::in | ios::binary);

        if (test_data_file->fail()) {
            delete test_data_file;
            test_data_file = NULL;
            throw invalid_argument("Unable to open test data file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,
            test_data_file->seekg(4); // skip 'magic number'
            test_data_items_count = file_read_big_endian_int32(*test_data_file);
            input_rows = file_read_big_endian_int32(*test_data_file);
            input_columns = file_read_big_endian_int32(*test_data_file);

            SPDLOG_DEBUG("count = " + to_string(test_data_items_count) + ", rows,cols = " + to_string(input_rows) + "," +
                         to_string(input_columns));

            // initialize test data bufferarray, first delete old one incase item count or bytes-per-item changes
            if (test_data_buffer != NULL) {
                for (int x = 0; x < test_data_items_count; x++) {
                    delete[] test_data_buffer[x];
                }
                delete[] test_data_buffer;
            }

            const int values_per_input = input_rows * input_columns;

            test_data_buffer = new float *[test_data_items_count];
            for (int x = 0; x < test_data_items_count; x++) {
                test_data_buffer[x] = new float[values_per_input];
            }
        }
    }

    /**
     * @brief Set the test labels file and verify its existence
     *
     * @param path
     */
    void set_test_labels_file(string path) {
        SPDLOG_INFO("Opening test labels file '" + path + "' ...");

        delete test_labels_file;

        test_labels_path = path;
        test_labels_file = new ifstream(path, ios::in | ios::binary);

        if (test_labels_file->fail()) {
            delete test_labels_file;
            test_labels_file = NULL;
            throw invalid_argument("Unable to open test labels file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            test_labels_file->seekg(4); // skip 'magic number'
            test_data_items_count = file_read_big_endian_int32(*test_labels_file);

            SPDLOG_DEBUG("count = " + to_string(test_data_items_count));

            // initialize training labels bufferarray, first delete old one incase batch size changes
            if (test_labels_buffer != NULL) {
                delete[] test_labels_buffer;
            }

            test_labels_buffer = new unsigned char[test_data_items_count];
        }
    }

    /**
     * @brief Throws error if specified file stream is NULL. We have a separate function for this incase we want to add
     *        additional logic and to keep wording between errors the same.
     */
    void verify_file_open(ifstream *filestream, string type) {
        if (filestream == NULL) {
            throw invalid_function_call(
                type + " file has not been opened for reading. Open file using setter functions in TrainingData class.");
        }
    }

    /**
     * @brief Get the next training batch from training data file
     */
    void get_next_training_data_batch() {
        verify_file_open(training_data_file, "Training data");

        int bytes_per_item = input_rows * input_columns;

        uint8_t temp; // temporarily stores byte to be converted to float

        for (int x = 0; x < batch_size; x++) {
            for (int y = 0; y < bytes_per_item; y++) {
                training_data_file->read((char *)(&temp), 1);
                training_data_batch_buffer[x][y] = temp / 255.0f; // normalize input between 0 and 1
                current_record++;
            }
            if (training_data_file->eof()) {
                break;
            }
        }

        current_batch++;

        // if we've reached the end of the training data file, re-open so that we loop back
        if (training_data_file->eof()) {
            SPDLOG_DEBUG("Looped back training data file");
            training_data_file->clear();
            training_data_file->seekg(16);
            current_record = 0;
            current_batch = 0;
        }
    }

    /**
     * @brief Get the next training labels batch from training labels data file
     */
    void get_next_training_labels_batch() {
        verify_file_open(training_labels_file, "Training labels");

        for (int x = 0; x < batch_size; x++) {
            training_labels_file->read((char *)(&training_labels_batch_buffer[x]), 1);
            if (training_labels_file->eof()) {
                break;
            }
        }

        // if we've reached the end of the training labels file, re-open so that we loop back
        if (training_labels_file->eof()) {
            SPDLOG_DEBUG("Looped back training labels file");
            training_labels_file->clear();
            training_labels_file->seekg(8);
            current_record = 0;
            current_batch = 0;
        }
    }

    /**
     * @brief Get the testing data from file
     */
    void get_test_data() {
        verify_file_open(test_data_file, "Test data");

        int bytes_per_item = input_rows * input_columns;

        uint8_t temp; // temporarily stores byte to be converted to float

        for (int x = 0; x < test_data_items_count; x++) {
            for (int y = 0; y < bytes_per_item; y++) {
                test_data_file->read((char *)(&temp), 1);
                test_data_buffer[x][y] = temp / 255.0f; // normalize input between 0 and 1
            }
        }
    }

    /**
     * @brief Get testing labels from file
     */
    void get_test_labels() {
        verify_file_open(test_data_file, "Test labels");

        for (int x = 0; x < test_data_items_count; x++) {
            test_labels_file->read((char *)(&test_labels_buffer[x]), 1);
        }
    }

    ~TrainingData() {
        delete training_data_file;
        delete training_labels_file;
        delete test_data_file;
        delete test_labels_file;

        if (training_data_batch_buffer != NULL) {
            for (int x = 0; x < batch_size; x++) {
                delete[] training_data_batch_buffer[x];
            }
        }
        delete[] training_data_batch_buffer;

        if (test_data_buffer != NULL) {
            for (int x = 0; x < test_data_items_count; x++) {
                delete[] test_data_buffer[x];
            }
        }
        delete[] test_data_buffer;

        delete[] test_labels_buffer;
        delete[] training_labels_batch_buffer;

        SPDLOG_DEBUG("Deleted training data");
    }
};