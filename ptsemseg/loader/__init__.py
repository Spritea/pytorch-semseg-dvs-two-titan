import json

from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import (
    MITSceneParsingBenchmarkLoader
)
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader

from ptsemseg.loader.my_loader import myLoader

import yaml
def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "my":myLoader,
    }[name]


def get_data_path(name, config_file=None):
    """get_data_path

    :param name:
    :param config_file:
    """
    # data = json.load(open(config_file))
    # with open(config_file) as fp:
    #     cfg = yaml.load(fp)
    # cfg=yaml.load(config_file)
    return config_file['data']['path']
