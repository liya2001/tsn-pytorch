import os


LIST_FILE_PATH = '/home/liya/workspace/trecvid/tsn-pytorch/data'


def get_list_file(mode, modality):
    return os.path.join(LIST_FILE_PATH, '{}_{}.txt'.format(mode, modality.lower()))
