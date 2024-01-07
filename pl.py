from viztools.sequence import WellSequence
import viztools as vz
import matplotlib.pyplot as plt
import cv2, copy
import numpy as np
import matplotlib.patches as mpatches
import math
import matplotlib.cm as cm
import matplotlib.animation as animation

def get_default_rc_params(dpi = 150):
    rc = {  'figure.dpi' : 150,
            'figure.figsize' : (4,4),
            'figure.frameon' : False,
            'axes.spines.left' : True,
            'axes.spines.right' : True,
            'axes.spines.top' : True,
            'axes.spines.bottom' : True,
            'axes.linewidth' : 2,
            'xtick.bottom' : False,
            'ytick.left' : False,
            'xtick.labelbottom' : False,
            'ytick.labelleft' : False
        }
    return rc

def plot_histogram(ws : WellSequence, ts = 0, channel = 'Brightfield'):
    img = ws.get_channel_timestep(channel, timesteps = ts)
    plt.hist(img.ravel(),256,[0,256]); plt.show()


def _get_nanowell_overlay(nanowells, shape):
    m = np.zeros(shape)

    for i, r in enumerate(nanowells):
        cv2.rectangle(m, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (255,255,255), 2)
        cv2.putText(m, str(i), (r[0] + int(r[2] / 3), r[1] + int(r[3] / 1.75)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255, 255),1)

    return m





def plot_frame(ws : WellSequence, ts = 0, channel = 'Brightfield',
                 dpi = 100, figsize = (4,4), title = None, overlay = 'all',
                 crop_coord = None, show_time = True, ax = None, animated = False,
                 interpolation = 'hanning'):



    minmax = lambda x : (x - x.min()) / (x.max() - x.min())

    if crop_coord:
        ws = copy.copy(ws)
        ws.set_X(ws.get_X()[:,:, crop_coord[0]:crop_coord[1],crop_coord[2]:crop_coord[3]])

    
    with plt.rc_context(rc=get_default_rc_params(dpi = dpi)):
        if not ax:
            fig, axes = plt.subplots(1,1, dpi = dpi, figsize = figsize)
        else:
            axes = ax

        axes.set_animated(animated)
        axes.imshow(ws.get_channel_timestep(channel, timesteps = ts), cmap = vz.cl.CM_DARKGRAY, interpolation = interpolation)
        if overlay == 'all':
            overlay = ['ontarget','bystander','t cell','death','nanowells']

        patches = []
        for o in overlay:
            if o == 'death':
                overlay_death = ws.get_channel_timestep(2, timesteps = ts)
                axes.imshow(overlay_death,cmap = vz.cl.CM_DEATH, vmin = 0, vmax = 255, interpolation = interpolation)
                patches.append(mpatches.Patch(color='red', label='Cell Death'))
            elif o == 'bystander':
                overlay_bystander = ws.get_channel_timestep(0, timesteps = ts)
                axes.imshow(overlay_bystander,cmap = vz.cl.CM_BYSTANDER, vmin = 0, vmax = 255, interpolation = interpolation)
                patches.append(mpatches.Patch(color='blue', label='Bystanders (CD19-)'))

            elif o == 'ontarget':
                overlay_ontarget = ws.get_channel_timestep(1, timesteps = ts)
                axes.imshow(overlay_ontarget,cmap = vz.cl.CM_ONTARGET, vmin = 0, vmax = 255, interpolation = interpolation)
                patches.append(mpatches.Patch(color='lime', label='Ontargets (CD19+)'))

            elif o == 't cell':
                overlay_tcell = ws.get_channel_timestep(3, timesteps = ts)
                axes.imshow(overlay_tcell,cmap = vz.cl.CM_TCELL, vmin = 0, vmax = 255, interpolation = interpolation)
                patches.append(mpatches.Patch(color='cyan', label='T Cells (CD8+)'))

            elif o == 'nanowells' and isinstance(ws.uns.get(o),np.ndarray):
                om = _get_nanowell_overlay(ws.uns.get(o), ws.shape)
                print(om.shape, om.min(), om.max())
                axes.imshow(om, cmap = vz.cl.CM_NANOWELLS)


            else:
                print(f"No key: {o} for overlay!")
        if title: axes.set_title(title)

        
        if patches:
            axes.legend(handles=patches, ncol = 4, fontsize = 5, bbox_to_anchor = (1.075, -0.05), frameon = False)

        if show_time:
            mm = ts * 8
            hh = math.floor(mm / 60)
            mm = mm % 60
            timestring = f"{str(hh).zfill(2)}:{str(mm).zfill(2)}"
            
            axes.text(0.85,0.05, timestring, transform = axes.transAxes, color = 'yellow')

    return axes





def to_mp4(ws, timesteps = None, fps = 4, overlay = 'all', 
    path = '/content/movie.mp4', dpi = 100, figsize = (2,2), interpolation = 'hanning'):
    frames = [] # for storing the generated images

    with plt.rc_context(rc = vz.pl.get_default_rc_params(dpi = dpi)):
        fig = plt.figure(dpi = dpi, figsize = figsize)

        if not timesteps: timesteps = (0, ws.timesteps)

        if overlay == 'all': overlay = ['bystander','ontarget','tcell','death']
        alphas = [0.0, 0.0, 0.0, 0.0]
        for o in overlay:
            if o == 'bystander': alphas[0] = 1.0
            if o == 'ontarget': alphas[1] = 1.0
            if o == 'tcell': alphas[2] = 1.0
            if o =='death': alphas[3] = 1.0
        
        BF_max = ws.get_channel('Brightfield').max()

        for i in range(timesteps[0], timesteps[1]):

            mm = i * 8
            hh = math.floor(mm / 60)
            mm = mm % 60
            timestring = f"{str(hh).zfill(2)}:{str(mm).zfill(2)}"
            interp = interpolation
            frames.append([plt.imshow(ws.get_channel_timestep('Brightfield', i), interpolation = interp, cmap=vz.cl.CM_DARKGRAY, animated=True, vmin = 0, vmax = BF_max, alpha = 1.0),
                        plt.imshow(ws.get_channel_timestep('Bystander', i), interpolation = interp, cmap=vz.cl.CM_BYSTANDER, animated=True, vmin = 0, vmax = 255, alpha = alphas[0]),
                        plt.imshow(ws.get_channel_timestep('Ontarget', i), interpolation = interp, cmap=vz.cl.CM_ONTARGET, animated=True, vmin = 0, vmax = 255, alpha = alphas[1]),
                        plt.imshow(ws.get_channel_timestep('T Cell', i), interpolation = interp, cmap=vz.cl.CM_TCELL, animated=True, vmin = 0, vmax = 255, alpha = alphas[2]),
                        plt.imshow(ws.get_channel_timestep('Death', i), interpolation = interp, cmap=vz.cl.CM_DEATH, animated=True, vmin = 0, vmax = 255, alpha = alphas[3]),
                        plt.text(0.975,0.05, timestring, transform = plt.gca().transAxes, va='center', ha = 'right', color = 'yellow')])

        ani = animation.ArtistAnimation(fig, frames, interval = int(1000 / fps), blit=True,repeat_delay=1000)
        ani.save(path, writer = "PillowWriter")


