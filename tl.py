import numpy as np
import cv2, math, os, sys
import viztools as vz
from viztools.sequence import WellSequence
from tqdm.notebook import tqdm as tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def get_colormap(cvals, colors):

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)
    cmap.set_under(alpha = 0)
    return cmap

# Returns an overlay matching dimension of img
def _get_bounding_rects(contours, img, min_size = 200, max_size = 25000):
    m = np.zeros(img.shape)

    contours = list(filter(lambda c : cv2.contourArea(c) > min_size, contours))
    contours = [c for c in contours if cv2.contourArea(c) < max_size]
    
    return [cv2.boundingRect(c) for c in contours]

def _find_nanowells(ws : WellSequence, threshold = 30):

    img = ws.get_channel_timestep('Brightfield', timesteps = 0)

    _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,)
    boundingRects = _get_bounding_rects(contours, img)
    boundingRects = cv2.groupRectangles(boundingRects, 1)[0]

    ws.uns['nanowells'] = boundingRects

def find_nanowells(ws : WellSequence, threshold = 30, do_sweep = False):
    if not do_sweep: 
        return _find_nanowells(ws, threshold)
    else:
        best_threshold = 0
        max_rects = 0
        for threshold in range(5,80,5):
            img = ws.get_channel_timestep('Brightfield', timesteps = 0)

            _, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,)
            boundingRects = _get_bounding_rects(contours, img)
            boundingRects = cv2.groupRectangles(boundingRects, 1)[0]

            n_rects = len(boundingRects)
            if n_rects > max_rects:
                best_threshold = threshold
                max_rects = n_rects

        
        print(f"Best threshold is: {best_threshold}")
        return _find_nanowells(ws, best_threshold)



def correlate_signals(ws : WellSequence):

    zscoreseries = lambda x : (x - x.mean(axis = (1,2), keepdims = True)) / x.std(axis = (1,2), keepdims = True)

    x1 = vz.cl.filter_img(ws.get_channel('Ontarget'), vz.cl.CM_ONTARGET)
    x2 = vz.cl.filter_img(ws.get_channel('Bystander'), vz.cl.CM_BYSTANDER)
    x3 = vz.cl.filter_img(ws.get_channel('Death'), vz.cl.CM_DEATH)
    x4 = vz.cl.filter_img(ws.get_channel('T Cell'), vz.cl.CM_TCELL)

    #x1 = zscoreseries(ws.get_channel('Ontarget'))
    #x2 = zscoreseries(ws.get_channel('Bystander'))
    #x3 = zscoreseries(ws.get_channel('Death'))
    #x4 = zscoreseries(ws.get_channel('T Cell'))

    signals = [x1,x2,x3,x4]
    corrs = []

    for s1 in signals:
        for s2 in signals:
            corrs.append((s1 * s2).mean(axis = (1,2)).mean())

    corrs = np.array(corrs)

    ws.uns['sxcorr'] = corrs


def get_mean_signals(ws : WellSequence):
    ws.uns['mean_brightfield'] = ws.get_channel('Brightfield').mean(axis = (1,2))
    ws.uns['mean_ontarget'] = ws.get_channel('Ontarget').mean(axis = (1,2))
    ws.uns['mean_bystander'] = ws.get_channel('Bystander').mean(axis = (1,2))
    ws.uns['mean_death'] = ws.get_channel('Death').mean(axis = (1,2))
    ws.uns['mean_tcell'] = ws.get_channel('T Cell').mean(axis = (1,2))



def collect_nanowell_data(ws : WellSequence):


    if len(ws.uns['nanowells']) == 0:
        print("No nanowells found!")
        return
    # Compute nanowell correlation and mean signals
    obs = []
    mtx = []
    obsm_bystander = []
    obsm_ontarget = []
    obsm_death = []
    obsm_tcell = []
    obsm_brightfield = []

    for nw in tqdm(map(ws.get_nanowell, range(len(ws.uns['nanowells']))), total = len(ws.uns['nanowells']), leave = False):
        vz.tl.correlate_signals(nw)
        vz.tl.get_mean_signals(nw)
        mtx.append(np.expand_dims(nw.uns['sxcorr'], axis = 0))

        obsm_bystander.append(np.expand_dims(nw.uns['mean_bystander'], axis = 0))
        obsm_ontarget.append(np.expand_dims(nw.uns['mean_ontarget'], axis = 0))
        obsm_death.append(np.expand_dims(nw.uns['mean_death'], axis = 0))
        obsm_tcell.append(np.expand_dims(nw.uns['mean_tcell'], axis = 0))
        obsm_brightfield.append(np.expand_dims(nw.uns['mean_brightfield'], axis = 0))
        
        obs.append([nw.uns['nw_idx']] + nw.uns['nw_crop_coords'].tolist())

    # Join together all correlations and mean signals
    nonan = lambda x : np.nan_to_num(x, nan = 0.0, posinf = 0.0, neginf = 0.0)

    ws.uns['obsm_sxcorr'] = nonan(np.concatenate(mtx, axis = 0))
    ws.uns['obsm_bystander'] = nonan(np.concatenate(obsm_bystander, axis = 0))
    ws.uns['obsm_ontarget'] = nonan(np.concatenate(obsm_ontarget, axis = 0))
    ws.uns['obsm_death'] = nonan(np.concatenate(obsm_death, axis = 0))
    ws.uns['obsm_tcell'] = nonan(np.concatenate(obsm_tcell, axis = 0))
    ws.uns['obsm_brightfield'] = nonan(np.concatenate(obsm_brightfield, axis = 0))

    ws.uns['obsm_combined_cell_signals'] = np.concatenate([np.expand_dims(ws.uns['obsm_bystander'].mean(axis = 1),1), 
                                            np.expand_dims(ws.uns['obsm_ontarget'].mean(axis = 1),1),
                                            np.expand_dims(ws.uns['obsm_tcell'].mean(axis = 1),1)], axis = 1)
    ws.uns['obsm_combined_cell_signals'] = pd.DataFrame(ws.uns['obsm_combined_cell_signals'], columns = ['bystander','ontarget','tcell'])

    ws.uns['obsm_combined_cell_signals'].index.name = 'nanowell'
    ws.uns['obsm_combined_cell_signals']['product'] = ws.uns['obsm_combined_cell_signals'].product(axis = 1)
import pandas as pd

def channel_cross_product(ws : WellSequence):
    channels = ['Ontarget', 'Bystander','T Cell', 'Death']
    channel_CM = {'Ontarget' : vz.cl.CM_ONTARGET,
                    'Bystander' : vz.cl.CM_BYSTANDER,
                    'T Cell' : vz.cl.CM_TCELL,
                    'Death' : vz.cl.CM_DEATH}
    sf = lambda s: s.lower().replace(' ','')
    cross_mtx = []
    for c1 in channels:
        for c2 in channels:
            #if c1 == c2: continue
            c1x = vz.cl.filter_img(ws.get_channel(c1), channel_CM.get(c1))
            c2x = vz.cl.filter_img(ws.get_channel(c2), channel_CM.get(c2))

            prod = c1x * c2x
            prod = prod.mean(axis = (1,2))
            name = f"{sf(c1)}_x_{sf(c2)}"
            cross_mtx.append(pd.Series(prod, name = name))
    cross_mtx = pd.concat(cross_mtx, axis = 1).T
    cross_mtx.columns.name = 'timestep'

    ws.uns['cross_mtx'] = cross_mtx

def identify_interesting_wells(ws : WellSequence, s = 2):
    mean_signal = ws.uns['obsm_combined_cell_signals']['product'].mean()

    intersting_nanowells = ws.uns['obsm_combined_cell_signals']['product'] > s * mean_signal
    return intersting_nanowells[intersting_nanowells].index.tolist()

