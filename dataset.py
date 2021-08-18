from utils import maybe_download
from collections import namedtuple
import tarfile
from xml.etree import ElementTree
from scipy import io
import numpy as np
from utils import to_categorical
import logging

from sklearn.model_selection import  train_test_split
from glob import glob

SOURCE_IMAGE_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
SOURCE_ANNOT_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar'
SOURCE_LISTS_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar'

DataSet = namedtuple('Data', ['files', 'labels', 'boxes'])


def _get_extract():
    # return the filepaths
    images = maybe_download('images.tar', SOURCE_IMAGE_URL)
    tar = tarfile.open(images)
    tar.extractall('dogImages/')
    tar.close()

    annotations = maybe_download('annotations.tar', SOURCE_ANNOT_URL)
    tar = tarfile.open(annotations)
    tar.extractall('dogImages/')
    tar.close()

    lists = maybe_download('lists.tar', SOURCE_LISTS_URL)
    tar = tarfile.open(lists)
    tar.extractall('dogImages/')
    tar.close()

    data_list = 'dogImages/file_list'
    train_list = 'dogImages/train_list'
    test_list = 'dogImages/test_list'

    return data_list, train_list, test_list


def _get_bbox_from_xml(xmlfile):
    root = ElementTree.parse(xmlfile).getroot()
    xmin = root.find('object').find('bndbox').find('xmin').text
    ymin = root.find('object').find('bndbox').find('ymin').text
    xmax = root.find('object').find('bndbox').find('xmax').text
    ymax = root.find('object').find('bndbox').find('ymax').text
    return int(xmin), int(ymin), int(xmax), int(ymax)


def _load_dataset(data_list):
    mat = io.loadmat(data_list)
    files = []
    labels = []
    annotations = []
    for f, l, a in zip(mat['file_list'], mat['labels'], mat['annotation_list']):
        files.append('dogImages/Images/' + f[0][0])
        labels.append(l[0] - 1)
        annotations.append(_get_bbox_from_xml('dogImages/Annotation/' + a[0][0]))
    files = np.array(files)
    labels = to_categorical(np.array(labels), 120)
    annotations = np.array(annotations)
    return files, labels, annotations


def load_data():
    logger = logging.getLogger(__name__)

    data_list, train_list, test_list = _get_extract()
    train_files, train_labels, train_bbox = _load_dataset(train_list)
    train_files, valid_files, train_labels, valid_labels, train_bbox, valid_bbox = train_test_split(train_files,
                                                                                                    train_labels,
                                                                                                    train_bbox)
    test_files, test_labels, test_bbox = _load_dataset(test_list)
    dog_names = [item[27:-1] for item in sorted(glob("dogImages/Images/*/"))]

    # print statistics about the dataset
    logger.info('There are %d total dog categories.' % len(dog_names))
    logger.info('There are %s total dog images.\n' % len(np.hstack([train_files, test_files])))
    logger.info('There are %d training dog images.' % len(train_files))
    logger.info('There are %d validation dog images.' % len(valid_files))
    logger.info('There are %d test dog images.' % len(test_files))

    train = DataSet(train_files, train_labels, train_bbox)
    valid = DataSet(valid_files, valid_labels, valid_bbox)
    test = DataSet(test_files, test_labels, test_bbox)

    return train, valid, test, dog_names