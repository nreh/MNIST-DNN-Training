#
# Note: This script is no longer used, however I'm keeping it here as it can be useful for other projects
#

# Python script for converting MNIST data to CSV files
# furthermore, it normalizes the values (0-255) to between 0 and 1.

import logging

logging.basicConfig(level=logging.DEBUG)

# 
# Binary format (From http://yann.lecun.com/exdb/mnist/)
#
# ===========================================================================================================================
#
# TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  60000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.

# TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  60000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

# TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
# 0004     32 bit integer  10000            number of items
# 0008     unsigned byte   ??               label
# 0009     unsigned byte   ??               label
# ........
# xxxx     unsigned byte   ??               label
# The labels values are 0 to 9.

# TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
# [offset] [type]          [value]          [description]
# 0000     32 bit integer  0x00000803(2051) magic number
# 0004     32 bit integer  10000            number of images
# 0008     32 bit integer  28               number of rows
# 0012     32 bit integer  28               number of columns
# 0016     unsigned byte   ??               pixel
# 0017     unsigned byte   ??               pixel
# ........
# xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

def convert_image_file(filename, destination):

    logging.info(f"Converting image file {filename} to {destination}")
    
    file = open(filename, 'rb')
    dest = open(destination, 'w')

    # MNIST dataset starts off with 4 byte magic number, so skip it
    file.seek(4)

    number_of_images = int.from_bytes(file.read(4), 'big')

    rows = int.from_bytes(file.read(4), 'big')          # should be 28
    columns = int.from_bytes(file.read(4), 'big')       # should be 28

    
    for i in range(number_of_images):
        for j in range(columns):
            for k in range(rows):
                dest.write(
                    str(int.from_bytes(file.read(1), 'big'))
                )
                if not (j == columns-1 and k == rows-1):
                    # commas between items until last row/column is reached
                    dest.write(',')
        
        if (i / number_of_images * 100) % 25 == 0:
            logging.info(f'{i/number_of_images * 100}%')

        dest.write('\n')

    logging.info('100% DONE!')

def convert_label_file(filename, destination):

    logging.info(f"Converting label file {filename} to {destination}")
    
    file = open(filename, 'rb')
    dest = open(destination, 'w')

    # MNIST dataset starts off with 4 byte magic number, so skip it
    file.seek(4)

    number_of_items = int.from_bytes(file.read(4), 'big')
    
    for i in range(number_of_items):
        label = int.from_bytes(file.read(1), 'big')

        for j in range(10):
            if label == j:
                dest.write('1')
            else:
                dest.write('0')
                    
            if not j == 9:
                # commas between items until last row/column is reached
                dest.write(',')
        
        if (i / number_of_items * 100) % 25 == 0:
            logging.info(f'{i/number_of_items * 100}%')

        dest.write('\n')
        
    logging.info('100% DONE!')



convert_image_file('bin/test-images.idx3-ubyte', 'test_images.csv')
convert_image_file('bin/train-images.idx3-ubyte', 'train_images.csv')

convert_label_file('bin/test-labels.idx1-ubyte', 'test_labels.csv')
convert_label_file('bin/train-labels.idx1-ubyte', 'train_labels.csv')