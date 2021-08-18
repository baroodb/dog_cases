import urllib.request as urlrequest
import os
import logging
import numpy as np


def maybe_download(file_name, url, data_path='./'):
    logger = logging.getLogger(__name__)

    file_path = data_path + file_name
    logger.debug(('Checking {} into {}'.format(file_name, data_path)))

    # Check data dir exists
    if not os.path.exists(data_path):
        logger.debug('Folder {} not found, creating it'.format(data_path))
        os.makedirs(data_path)

    # Check data file exists
    if os.path.exists(file_path):
        logger.debug('File {} found'.format(file_path))
        return file_path

    # Otherwise download it
    logger.info('Downloading file {} from {}'.format(file_path, url))
    temp_file_name, _ = urlrequest.urlretrieve(url, file_path)
    logger.info('Successfully downloaded file {}, {} bites'.format(temp_file_name, os.stat(temp_file_name).st_size))

    return file_path


def to_categorical(y, num_classes=None):

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
