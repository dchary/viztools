#import czifile
import numpy as np
import viztools as vz
import re
from viztools.sequence import WellSequence

def create_test_data(frames = 10, channels = 5, shape = (256,256)):
    return np.zeros((frames, channels, shape[0], shape[1]), dtype = np.uint8)


def read_czi(p : str, *args, **kwargs):
    img = czifile.imread(p, *args, **kwargs)
    img = np.squeeze(img)

    ws = vz.sequence.WellSequence(img)
    ws.uns['filepath'] = p
    return ws

def parse_well_num(p : str):
    res = re.findall("[wW]ell [0-9]+", p)
    if res:
        return int(res[0].split(" ")[1])

def parse_well_scene(p : str):
    res = re.findall("Scene-[0-9]+-P[0-9]+", p)
    if res:
        return res[0].split("-")[1:]

def parse_well_date(p : str):
    res = re.findall("[0-9]+-[0-9]+-[0-9]+", p)
    if res:
        return [int(x) for x in res[0].split("-")]

def get_full_name(ws : WellSequence, nw_idx = None):
    well = ws.uns['metadata']['well_num']
    scene1,scene2 = ws.uns['metadata']['scene']
    month, day, year = ws.uns['metadata']['date']
    fname = f"date-{month}-{day}-{year}_well-{well}_scene-{scene1}-{scene2}"
    if nw_idx: fname += f"_nw-{nw_idx}"
    return fname
