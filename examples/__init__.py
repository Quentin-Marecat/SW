## matplotlib setting
from matplotlib import colors
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
from PIL import Image, ImageChops

ICGMmarine=(0.168,0.168,0.525)
ICGMblue=(0,0.549,0.714)
ICGMorange=(0.968,0.647,0)
ICGMyellow=(1,0.804,0)
gray=(0.985,0.985,0.985)
map1=tuple(np.array([255, 195, 15])/256)
map2=tuple(np.array([255, 87, 51])/256)
map3=tuple(np.array([199, 0, 57])/256)
map4=tuple(np.array([144, 12, 63])/256)
map5=tuple(np.array([84, 24, 69])/256)
clr_map=[gray,ICGMyellow,ICGMorange,ICGMblue,ICGMmarine]
clr = [ICGMmarine,ICGMorange,ICGMblue,ICGMyellow]
cmap = colors.LinearSegmentedColormap.from_list('my_list', clr_map, N=100)
default_color=ICGMmarine

#print(plt.matplotlib.rcParams.keys())
plt.matplotlib.rcParams.update({'figure.figsize': (12, 10),'figure.autolayout': False,'font.sans-serif':'DejaVu Sans',\
    'font.size':30,'lines.linewidth':2.5,'lines.markersize':0.01,'lines.marker':'*','image.cmap':cmap,\
    'axes.prop_cycle':cycler(color=clr), 'savefig.format':'png',\
    'axes.grid.axis': 'both','grid.color':'k','grid.linestyle':'--',\
    'text.color':default_color,'axes.labelcolor':default_color,'xtick.color':default_color,'ytick.color':default_color,'grid.color':default_color,\
    'boxplot.boxprops.color':default_color,'hatch.color': default_color,'axes.edgecolor': default_color,})



def crop(im,filename=None):
    if filename==None:
        try:
            filename=im.filename
        except AttributeError:
            filename=f'noname.{plt.matplotlib.rcParams["savefig.format"]}'
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        im_crop = im.crop(bbox)
        im_crop.save(filename)
    else: 
        # Failed to find the borders, convert to "RGB"        
        return crop(im.convert('RGB'),im.filename)