# ip_utils.py
# Simple image processing utilities.
#
# Bob Dougherty <bobd@stanford.edu>
# released under the WTFPL licence (http://www.wtfpl.net/)

import numpy as np
from scipy import ndimage as ndimage

def montage(vol, ncols=None):
    """Returns a 2d image monage given a 3d volume."""
    ncols = ncols if ncols else int(np.ceil(np.sqrt(vol.shape[2])))
    rows = np.array_split(vol, range(ncols,vol.shape[2],ncols), axis=2)
    # ensure the last row is the same size as the others
    rows[-1] = np.dstack((rows[-1], np.zeros(rows[-1].shape[0:2] + (rows[0].shape[2]-rows[-1].shape[2],))))
    im = np.vstack([np.squeeze(np.hstack(np.dsplit(row, ncols))) for row in rows])
    return(im)

def montage_4d(vol):
    return np.hstack([montage(vol[...,i], 1) for i in range(vol.shape[3])])

def std_unbiased(std, n):
    """ return an unbiased stdev estimate (see http://nbviewer.ipython.org/5548327) """
    from scipy.special import gamma
    bias = std * (1. - np.sqrt(2. / (n-1)) * (gamma(n / 2.) / gamma((n-1.) / 2.)))
    return std + bias

def get_mask(image_array, mask_thresh=None):
    image_array = image_array.astype(np.float)
    if not mask_thresh:
        mask_thresh = np.percentile(image_array, 20.0)
    mask = ndimage.median_filter(image_array, 3) > mask_thresh
    mask = ndimage.gaussian_filter(mask.astype(float), 0.2)
    mask = ndimage.grey_erosion(mask, size=(5,5,5))
    mask = ndimage.grey_dilation(mask, size=(5,5,5))
    mask = mask>0.5
    mask,n = ndimage.measurements.label(mask)
    obj_label = np.percentile(mask[mask>0],51)
    mask = mask==obj_label
    mask = ndimage.morphology.binary_fill_holes(mask)
    return mask

def autocrop(vol, mask=None, pad=3):
    '''Crops a 3d numpy array to within pad voxels of non-zero data. If mask is None, then vol is used as the mask.'''
    if mask==None:
        mask = vol
    xlim = np.argwhere(mask.sum(axis=2).sum(axis=1))[[0,-1]].squeeze()
    xlim = max(0,xlim[0]-pad),min(vol.shape[0],xlim[1]+pad)
    ylim = np.argwhere(mask.sum(axis=2).sum(axis=0))[[0,-1]].squeeze()
    ylim = max(0,ylim[0]-pad),min(vol.shape[1],ylim[1]+pad)
    zlim = np.argwhere(mask.sum(axis=1).sum(axis=0))[[0,-1]].squeeze()
    zlim = max(0,zlim[0]-pad),min(vol.shape[2],zlim[1]+pad)
    vol = vol[xlim[0]:xlim[1], ylim[0]:ylim[1], zlim[0]:zlim[1]]
    return vol

def align(src, tgt, src2=None):
    '''
    Align src image to tgt image using nipy's histogram registration algotrithm.
    Optionally pull src2 along for the ride and concatenate along axis=3.
    Images should be nipy image-like objects (e.g., as returned by nibabel.load).
    Note that only image data are returned; the affine of the aligned data is
    identical to that of the target image.
    '''
    import nipy.algorithms.registration
    reg = nipy.algorithms.registration.HistogramRegistration(src, tgt)
    T = reg.optimize('rigid')
    It = nipy.algorithms.registration.resample(src, T.inv(), reference=tgt, interp_order=7)
    if src2!=None:
        It2 = []
        for s in src2:
            It2.append(nipy.algorithms.registration.resample(s, T.inv(), reference=tgt, interp_order=7).get_data()[...,None])
        return It.get_data(),np.concatenate(It2, axis=3)
    else:
        return It.get_data()

def clip(im, hmax=98, hmin=30):
    l,u = np.percentile(im,[hmin,hmax])
    im = np.clip(im, l, u)
    im = (im-l) / (u-l)
    return im

def autoclip(vol, upper_pctile=95.0,max_scale=None):
    thresh = np.percentile(vol, upper_pctile)
    vol[vol>thresh] = thresh;
    if max_scale!=None:
        vol = vol/vol.max()*max_scale
    return vol

def add_subplot_axes(fig, ax, rect, axisbg='w', label_size=None):
    # E.g., to add a custom colorbar:
    # c = np.vstack((np.linspace(0,1.,N), np.linspace(1,0,N), np.ones((2,N)))).T
    # cbax = add_subplot_axes(fig, ax2, [.85,1.11, 0.25,0.05])
    # plt.imshow(np.tile(c,(2,1,1)).transpose((0,1,2)), axes=cbax)
    # cbax.set_yticks([])
    # cbax.set_xlabel('Slice number')
    # plt.tight_layout()
    # plt.savefig(outfile, bbox_inches='tight')
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    if label_size==None:
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
    else:
        x_labelsize = label_size
        y_labelsize = label_size
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

def xform_coord(xform, coord):
    c = (np.matrix(xform) * np.hstack((np.matrix(coord), np.matrix([1]))).T)
    return np.array(c[:3].T)[0]

def hstack_centered(images):
    num_ims = len(images)
    h = np.array([im.shape[0] for im in images])
    w = np.array([im.shape[1] for im in images])
    y_off = (h.max()-h)/2
    if len(images[0].shape)>2 and images[0].shape[2]==3:
        im = np.zeros((h.max(),w.sum(),3), dtype=images[0].dtype)
        for i in xrange(num_ims):
            x_off = w[:i].sum()
            im[y_off[i]:y_off[i]+h[i], x_off:x_off+w[i], :] = images[i]
    else:
        im = np.zeros((h.max(),w.sum()), dtype=images[0].dtype)
        for i in xrange(num_ims):
            x_off = w[:i].sum()
            im[y_off[i]:y_off[i]+h[i], x_off:x_off+w[i]] = images[i]
    return im

def three_axis(d, sl=None):
    if not sl:
        sl = [x/2 for x in d.shape[::-1]]
    ax = np.rot90(d[:,:,sl[0]])
    cr = np.rot90(d[:,sl[1],:])
    sg = np.fliplr(np.rot90(d[sl[2],:,:]))
    im = hstack_centered((ax,cr,sg))
    return im

def prep_brain(ni_file, clip=None):
    import nibabel as nb

    try:
        ni = nb.as_closest_canonical(nb.load(ni_file))
    except:
        ni = nb.as_closest_canonical(ni_file)
    d = np.copy(ni.get_data())
    if clip:
        try:
            d = d.clip(clip[0],clip[1])
        except:
            d = autoclip(d,clip)
            clip = [d.min(), d.max()]
    else:
        clip = [d.min(), d.max()]
    return d,clip

def plot_brain(m, cmap, clip, txt_color='w', outfile=None, label=None, fig=None, label_size=None, overlay=None, overlay_cmap='autumn', overlay_clip=None):
    from matplotlib import cm
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.figure(figsize=(16,6))
    im = plt.imshow(m, cmap=cmap, vmin=clip[0], vmax=clip[1])
    if overlay:
        if not overlay_clip:
           overlay_clip = [overlay.min(),overlay.max()]
        ol = plt.imshow(m, cmap=overlay_cmap, vmin=overlay_clip[0], vmax=overlay_clip[1])
    plt.axis('off')
    if label:
        # colorbar
        if label_size==None:
            label_size = [.30,0.07, 0.30,0.08]
        c = getattr(cm,cmap)
        cmap = np.array([c(i)[0:3] for i in xrange(c.N)])
        cbax = add_subplot_axes(fig, im.axes, label_size)
        plt.imshow(np.tile(cmap,(12,1,1)).transpose((0,1,2)), axes=cbax)
        cbax.set_yticks([])
        cbax.set_xlabel(label, color=txt_color, size=18, labelpad=0)
        cbax.set_xticks(np.linspace(0,255,5))
        if isinstance(clip[0], (int,long)):
            cbax.set_xticklabels(['%d'%v for v in np.linspace(clip[0],clip[1],5)])
        else:
            cbax.set_xticklabels(['%0.1f'%v for v in np.linspace(clip[0],clip[1],5)])
        cbax.xaxis.set_tick_params(labelsize=16, labelcolor=txt_color)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, transparent=True, dpi=150)

def compare_brains(ni_files, sl, outfile=None, clip=None, label=None, cmap='gray', fig=None, label_size=None):
    ims = []
    txt_color = 'w'
    for i in xrange(len(ni_files)):
        d,clip = prep_brain(ni_files[i], clip)
        if len(d.shape)>3 and d.shape[3]==3:
            mask = d.sum(axis=3)
            r = np.rot90(autocrop(d[...,0], mask, pad=0)[:,sl[i],:])
            g = np.rot90(autocrop(d[...,1], mask, pad=0)[:,sl[i],:])
            b = np.rot90(autocrop(d[...,2], mask, pad=0)[:,sl[i],:])
            ims.append(np.dstack((r,g,b)).astype(np.uint8))
            cmap = None
        else:
            ims.append(np.rot90(autocrop(d, pad=0)[:,sl[i],:]))
    im = hstack_centered(ims)
    plot_brain(im, cmap, clip, txt_color, outfile, label, fig, label_size)

def show_brain(ni_file, sl=None, outfile=None, clip=None, label=None, cmap='gray', fig=None, bg='', overlay_file=None, overlay_cmap='autumn', overlay_clip=None):
    from scipy import ndimage

    d,clip = prep_brain(ni_file, clip)
    txt_color = 'w'
    if len(d.shape)>3 and d.shape[3]==3:
        mask = d.sum(axis=3)
        r = three_axis(autocrop(d[...,0], mask),sl)
        g = three_axis(autocrop(d[...,1], mask),sl)
        b = three_axis(autocrop(d[...,2], mask),sl)
        m = np.dstack((r,g,b)).astype(np.uint8)
        cmap = None
    else:
        if 't' in bg:
            m0 = ndimage.morphology.binary_fill_holes(d>0)
            m1 = m0
            for i in xrange(2):
                m1 = ndimage.binary_erosion(m1)
                edge = np.logical_xor(m1,m0)
                d[edge] = d[edge]+(clip[1]-d[edge])*.5
        m = three_axis(autocrop(d, pad=0), sl)
        if 'w' in bg:
            m[m==0] = m.max()
            txt_color = 'k'
    if overlay_file:
        od,junk = prep_brain(overlay_file)
        overlay = three_axis(autocrop(od, mask=d, pad=0), sl)
        overlay = np.ma.masked_where(np.logical_and(overlay>=overlay_clip[0], overlay<=overlay_clip[1]), overlay)
    else:
        overlay = None
    plot_brain(m, cmap, clip, txt_color, outfile, label, fig)

