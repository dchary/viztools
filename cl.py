import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
def get_colormap(cvals, colors, alphas = None, max_alpha = 0.33):


    if not alphas:
        alphas = np.ones(len(colors)) * max_alpha
        alphas[0:2] = 0

    colors = list(map(mpl.colors.to_rgba, colors))
    
    # set alphas
    for i in range(len(colors)):
        colors[i] = (colors[i][0], colors[i][1], colors[i][2], alphas[i])
        

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)
    cmap.set_under(alpha = 0)
    return cmap

def get_default_colormap(basecolor, cvals = [0,60, 175,255], max_alpha = 0.5):
    return get_colormap(cvals = cvals, colors = ['black', 'black', basecolor, basecolor], max_alpha = max_alpha)

CM_BYSTANDER = get_default_colormap('blue', cvals = [0, 50, 100, 255], max_alpha = 0.3)
CM_ONTARGET = get_default_colormap('lime', cvals = [0,50,100,255], max_alpha = 0.5)
CM_TCELL = get_default_colormap('cyan', cvals = [0,10,50,255], max_alpha = 0.6)
CM_DEATH = get_default_colormap('crimson', cvals = [0,8,50,255], max_alpha = 0.6)
CM_DARKGRAY = get_colormap([0,225,255], ['black','gray','white'],[1.0,1.0,1.0])
CM_NANOWELLS = get_colormap([0,10,255], ['black','orange','orange'], [0.0,0.25,0.25], max_alpha = 0.25)


def rgba_to_grey(img):
    return np.mean(img[...,0:3], axis = -1) * img[...,-1]

def filter_img(img, CM):
    return rgba_to_grey(CM(img))
