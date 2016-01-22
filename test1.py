# test1 library for jupyter notebooks.

# import necessary packages
import numpy as np
from scipy import misc
from scipy import ndimage
import matplotlib.patches as patches
from matplotlib import pyplot as plt
from math import ceil

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def taxicab_diagonal(self):
        '''
        Return the taxicab distance from (x1,y1) to (x2,y2)
        '''
        return self.x2 - self.x1 + self.y2 - self.y1
    def overlaps(self, other):
        '''
        Return True iff self and other overlap.
        '''
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))
    def __eq__(self, other):
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)

def show_plots(plots, cols=2, x=10, y=5, bboxes = []):
    plt.figure(figsize=(x,y))
    i = 1
    c = len(plots)
    for plot in plots:
        ax = plt.subplot(ceil(c/2.), cols, i)
        try:
            if len(bboxes) > 0:
                for bbox in bboxes[i-1]:
                    xwidth = bbox.x2 - bbox.x1
                    ywidth = bbox.y2 - bbox.y1
                    p = patches.Rectangle((bbox.x1, bbox.y1), xwidth, ywidth,
                                          fc = 'none', ec = 'red')
                    ax.add_patch(p)
            plt.imshow(plot)
        except TypeError as e:
            print (e)
        except ValueError as e:
            print (e)
        plt.axis('off')
        i += 1
    plt.show()

def half_size(A, amount=50, interp='bicubic', mode=None):
    """ nearest, bilinear, bicubic, cubic """
    return misc.imresize(A, amount, interp, mode)

def label(im):
    mask = im > im.mean()
    return ndimage.label(mask), mask

def process(C, n=9, amount=50, interpolation='bicubic', mode=None):
    plots = []
    labels = []
    plots.append(C)
    labels.append(label(C))
    for x in range(n-1):
        C = half_size(C, amount, interpolation, mode)
        plots.append(C)
        labels.append(label(C))
    return labels, plots

def largest_labels(original_labels, images, n = 1, s  = 1000):
    label, mask = original_labels[n]
    label_im, nb_labels = label
    # Find the largest connect component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < s
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    _label_im = np.searchsorted(np.unique(label_im), label_im)
    # Now that we have only one connect component, extract it's bounding box
    rois = []
    try:
        for slice_x, slice_y in ndimage.find_objects(_label_im):
            rois.append(images[n][slice_x, slice_y])
    except IndexError as e:
        print (e)
    except ValueError as e:
        print (e)
    return rois

def labels_to_bboxes(original_labels, images, s  = 1000):
    n=0;
    bboxes = []
    for image in images:
        label, mask = original_labels[n]
        label_im, nb_labels = label
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        mask_size = sizes < s
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
        _label_im = np.searchsorted(np.unique(label_im), label_im)
        data_slices = ndimage.find_objects(_label_im)
        bbox = slice_to_bbox(data_slices)
        bboxes.append(bbox)
        n+=1
    return bboxes

def slice_to_bbox(slices):
    for s in slices:
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)
