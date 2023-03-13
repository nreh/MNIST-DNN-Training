#pragma once

#include <fstream>

/**
 * @brief Reads 4 byte big endian integer from file and returns an integer respectful of the system's endianness
 *
 * @return Big endian parsed integer
 */
int file_read_big_endian_int32(std::ifstream &stream) {

    int num = 1;
    if (*(char *)&num == 1) {
        // the system is little endian, data needs to be loaded in backwards
        const int byte_count = 4; // int32 contains 4 bytes

        // we use union to convert between char array and int32
        union {
            int32_t int32;
            char bytes[byte_count];
        } data;

        // load values in reverse order from file
        for (int x = byte_count - 1; x >= 0; x--) {
            stream.read(&data.bytes[x], 1);
        }

        return data.int32;
    } else {
        // the system is big endian, so we can load in data normally
        int data;
        stream.read((char *)&data, 4);
        return data;
    }
}