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
private:
    /**
     * @brief Number of items in single batch
     */
    int batch_size = 100;

public:

    // File readers

    ifstream* training_data_file = NULL;
    ifstream* training_labels_file = NULL;
    ifstream* test_data_file = NULL;
    ifstream* test_labels_file = NULL;

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
     * @brief 2-D array containing training batch data. Because batch size doesn't change during training, we initialize
     *        this before hand so we don't need to reallocate memory every batch.
     */
    float** training_data_batch_buffer = NULL;

    /**
     * @brief 1-D array containing training batch labels. Because batch size doesn't change during training, we initialize
     *        this before hand so we don't need to reallocate memory every batch.
     */
    unsigned char* training_labels_batch_buffer = NULL;

    /**
     * @brief 2-D array containing test data.
     */
    float** test_data_buffer = NULL;

    /**
     * @brief 1-D array containing test data labels.
     */
    unsigned char* test_labels_buffer = NULL;

    /**
     * @brief Set the training input file and verify its existence
     *
     * @param path
     */
    void set_training_data_file(string path) {
        training_data_path = path;
        training_data_file = new ifstream(path, ios::in | ios::binary);

        if (training_data_file->fail()) {
            delete training_data_file;
            training_data_file = NULL;
            throw invalid_argument("Unable to open training input file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            training_data_file->seekg(4); // skip 'magic number'
            training_data_file->read((char*)(&training_data_items_count), 4);
            training_data_file->read((char*)(&input_rows), 4);
            training_data_file->read((char*)(&input_columns), 4);
        }
    }

    /**
     * @brief Set the training labels file and verify its existence
     *
     * @param path
     */
    void set_training_labels_file(string path) {
        training_labels_path = path;
        training_labels_file = new ifstream(path, ios::in | ios::binary);

        if (training_labels_file->fail()) {
            delete training_labels_file;
            training_labels_file = NULL;
            throw invalid_argument("Unable to open training labels file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            training_data_file->seekg(4); // skip 'magic number'
            training_data_file->read((char*)(&training_data_items_count), 4);
        }
    }

    /**
     * @brief Set the test input file and verify its existence
     *
     * @param path
     */
    void set_test_data_file(string path) {
        test_data_path = path;
        test_data_file = new ifstream(path, ios::in | ios::binary);

        if (test_data_file->fail()) {
            delete test_data_file;
            test_data_file = NULL;
            throw invalid_argument("Unable to open test data file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            training_data_file->seekg(4); // skip 'magic number'
            training_data_file->read((char*)(&test_data_items_count), 4);
            training_data_file->read((char*)(&input_rows), 4);
            training_data_file->read((char*)(&input_columns), 4);
        }
    }

    /**
     * @brief Set the test labels file and verify its existence
     *
     * @param path
     */
    void set_test_labels_file(string path) {
        test_labels_path = path;
        test_labels_file = new ifstream(path, ios::in | ios::binary);

        if (test_labels_file->fail()) {
            delete test_labels_file;
            test_labels_file = NULL;
            throw invalid_argument("Unable to open test labels file '" + path + "'");
        } else {
            // succeeded in opening file, read metadata describing format of data,

            training_data_file->seekg(4); // skip 'magic number'
            training_data_file->read((char*)(&test_data_items_count), 4);
        }
    }

    /**
     * @brief Throws error if specified file stream is NULL. We have a separate function for this incase we want to add
     *        additional logic and to keep wording between errors the same.
     */
    void verify_file_open(ifstream* filestream, string type) {
        if (filestream == NULL) {
            throw invalid_function_call(
                type + " file has not been opened for reading. Open file using setter functions in TrainingData class."
            );
        }
    }

    /**
     * @brief Get the next training batch from training data file
     *
     * @param data 2-D array that training batch will be written to. Should already be initialized.
     *
     * @return Array containing training batch obtained from file
     */
    void get_next_training_data_batch(float** data) {
        verify_file_open(training_data_file, "Training data");

        int bytes_per_item = input_rows * input_columns;

        for (int x = 0; x < batch_size; x++) {
            for (int y = 0; y < bytes_per_item; y++) {
                training_data_file->read((char*)(&training_data_batch_buffer[x][y]), 1);
                current_record++;
            }
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
            training_labels_file->read((char*)(&training_labels_batch_buffer[x]), 1);
        }
    }

    /**
     * @brief Get the testing data from file
     *
     * @param data 2-D array that testing data will be written to. Should already be initialized.
     */
    void get_test_data(float** data) {
        verify_file_open(test_data_file, "Test data");

        int bytes_per_item = input_rows * input_columns;

        for (int x = 0; x < batch_size; x++) {
            for (int y = 0; y < bytes_per_item; y++) {
                test_data_file->read((char*)(&test_data_buffer[x][y]), 1);
            }
        }
    }

    /**
     * @brief Get testing labels from file
     *
     * @param data 2-D array that items will be written to. Should already be initialized.
     * @param item_count How many items to read from file
     */
    void get_test_labels(float** data) {
        verify_file_open(test_data_file, "Test labels");

        for (int x = 0; x < batch_size; x++) {
            test_labels_file->read((char*)(&test_labels_buffer[x]), 1);
        }
    }

    ~TrainingData() {
        delete training_data_file, training_labels_file, test_data_file, test_labels_file;
        delete[] test_data_buffer, test_labels_buffer, training_data_batch_buffer, training_labels_batch_buffer;
    }

};