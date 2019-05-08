from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .aicity666 import aiCity666_raw
from .aicity666 import aiCity666_vp 
from .aicity666 import aiCity666_veri 

__imgreid_factory = {
    'aicity666_vp':aiCity666_vp,
    'aicity_raw':aiCity666_raw,
    'aicity_veri':aiCity666_veri
}


def get_names():
    return list(__imgreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)
