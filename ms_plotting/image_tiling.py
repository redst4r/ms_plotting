import numpy as np


def tile_raster_images(X, img_dim, tile_shape=None, tile_spacing=(1, 1),
                       scale_rows_to_unit_interval=False, cmap=None):
    assert X.ndim == 3, "X must be 3D"
    import matplotlib.pyplot as plt
    if cmap is None:
        cmap = plt.cm.Greys_r

    shape = X.shape

    sample_dim = list(set(range(len(shape))) - set(img_dim))[0]  # just the remaining dimension
    n_sample = shape[sample_dim]

    if tile_shape is None:
        tiles_x = int(np.ceil((np.sqrt(n_sample))))
        tiles_y = 1 + n_sample // tiles_x
        tile_shape = (tiles_x, tiles_y)

    xbig = arange_in_tiles(X, img_dim=img_dim, tile_shape=tile_shape,
                           scale_rows_to_unit_interval=scale_rows_to_unit_interval, tile_spacing=tile_spacing)

    plt.imshow(xbig, cmap)


def arange_in_tiles(X, img_dim, tile_shape, tile_spacing=(1, 1), scale_rows_to_unit_interval=False):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.
    """

    assert img_dim
    assert max(img_dim) < X.ndim, "X has only %d dimensions, you specified spatial dimensions %d,%d." % (
    X.ndim, img_dim[0], img_dim[1])

    """
    the entire code below assumes that channels are in the first dimension, then space, i.e. (c,x,y)
    lets just make it that way
    """
    channel_dim = [_ for _ in range(3) if _ not in img_dim]
    assert len(channel_dim) == 1
    channel_dim = channel_dim[0]
    X = X.transpose([channel_dim, img_dim[0], img_dim[1]])

    # now space is in dim 1,2
    img_shape = X.shape[1:]

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape = [0,0]
    # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]

    # if we are dealing with only one channel
    H, W = img_shape
    Hs, Ws = tile_spacing

    # generate a matrix to store the output
    out_array = np.zeros(out_shape, dtype=X.dtype)

    # make the entire ouput between 0,1.  the FLAG scale_rows_to_unit_interval does this for each sample independently!!

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                if scale_rows_to_unit_interval:
                    # if we should scale values to be between 0 and 1
                    # do this by calling the `scale_to_unit_interval`
                    # function
                    this_img = _scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                else:
                    this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                # add the slice to the corresponding position in the
                # output array
                out_array[
                tile_row * (H + Hs): tile_row * (H + Hs) + H,
                tile_col * (W + Ws): tile_col * (W + Ws) + W
                ] \
                    = this_img

    return out_array

def tile_raster_RGB(X, tile_shape, scale_rows_to_unit_interval):
    import matplotlib.pyplot as plt

    assert X.shape[3] == 3, 'works only for three channels'
    assert len(tile_shape) == 2
    img_shape = X.shape[1:3]

    R = arange_in_tiles(X[...,0], img_dim=[0, 1], tile_shape= tile_shape, scale_rows_to_unit_interval=scale_rows_to_unit_interval)
    G = arange_in_tiles(X[...,1], img_dim=[0, 1], tile_shape= tile_shape, scale_rows_to_unit_interval=scale_rows_to_unit_interval)
    B = arange_in_tiles(X[...,2], img_dim=[0, 1], tile_shape= tile_shape, scale_rows_to_unit_interval=scale_rows_to_unit_interval)
    rgbI = np.stack([R, G, B], axis=-1)
    plt.imshow(rgbI)
    return rgbI

"""
def tile_raster_RGB(X, tile_shape, tile_spacing, scale_rows_to_unit_interval=False):
    assert X.shape[3] == 3, 'works only for three channels'
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2
    img_shape = X.shape[1:3]

    H, W = img_shape
    Hs, Ws = tile_spacing

    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                 in zip(img_shape, tile_shape, tile_spacing)]
    out_array = np.zeros(out_shape + [3], dtype=X.dtype)

    X = _scale_to_unit_interval(X)

    for tile_row in range(tile_shape[0]):
        for tile_col in range(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                if scale_rows_to_unit_interval:
                    # if we should scale values to be between 0 and 1
                    # do this by calling the `scale_to_unit_interval`
                    # function
                    this_img = _scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col])
                else:
                    this_img = X[tile_row * tile_shape[1] + tile_col]
                # add the slice to the corresponding position in the
                # output array
                out_array[tile_row * (H + Hs): tile_row * (H + Hs) + H, tile_col * (W + Ws): tile_col * (W + Ws) + W, :] \
                    = this_img
    plt.figure()
    plt.imshow(out_array, interpolation='nearest')
"""

def tile_raster_multicolor(I, tile_shape):
    import matplotlib.pyplot as plt

    channels = I.shape[-1]
    # layout the channels into triplets as an tiled-RGB image
    Rchan = arange_in_tiles(I[:, :, range(0, channels, 3)], img_dim=[0, 1], tile_shape=tile_shape)
    Gchan = arange_in_tiles(I[:, :, range(1, channels, 3)], img_dim=[0, 1], tile_shape=tile_shape)
    Bchan = arange_in_tiles(I[:, :, range(2, channels, 3)], img_dim=[0, 1], tile_shape=tile_shape)
    rgbI = np.stack([Rchan, Gchan, Bchan], axis=-1)
    rgbI -= rgbI.min()
    rgbI /= rgbI.max()

    plt.imshow(rgbI)
    return rgbI


def _scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= np.nanmin(ndar)
    ndar *= 1.0 / (np.nanmax(ndar) + eps)
    return ndar


R = np.array([255, 0, 0])
G = np.array([0, 255, 0])
B = np.array([0, 0, 255])


def gray_scale(img):
    '''Converts the provided RGB image to gray scale.
    '''
    # img = np.average(img, axis=2)
    assert img.ndim == 2
    return np.transpose([img, img, img], axes=[1, 2, 0])


def normalize(attrs, ptile=99):
    '''Normalize the provided attributions so that they fall between
       -1.0 and 1.0.
    '''
    h = np.percentile(attrs, ptile)
    l = np.percentile(attrs, 100 - ptile)
    return np.clip(attrs / max(abs(h), abs(l)), -1.0, 1.0)


def visualize_attrs_overlay(img, attrs, pos_ch=G, neg_ch=R, ptile=99):
    '''Visaualizes the provided attributions by first aggregating them
     along the color channel and then overlaying the positive attributions
     along pos_ch, and negative attributions along neg_ch.

     positive: green
     negative: red
    '''
    attrs = gray_scale(attrs)
    attrs = normalize(attrs, ptile)
    pos_attrs = attrs * (attrs >= 0.0)
    neg_attrs = -1.0 * attrs * (attrs < 0.0)
    attrs_mask = pos_attrs * pos_ch + neg_attrs * neg_ch

    attrs_mask -= attrs_mask.min()
    attrs_mask /= attrs_mask.max()

    img = gray_scale(img)
    img -= img.min()
    img /= img.max()

    vis = 0.5 * img + 0.5 * attrs_mask
    return vis


def overlay_grey_and_color(Xgrey, Xcolor):
    """"
    overlay a grayscale image with some color (e.g. segmentation masks)

    note that this IGNORES the magnitude of the color image, essentially it becomes a MASK
    :param Xgrey:
    :param Xcolor:
    :return:
    """
    from skimage import color, io, img_as_float
    import numpy as np
    import matplotlib.pyplot as plt

    alpha = 0.6

    img = img_as_float(Xgrey)
    rows, cols = img.shape

    color_mask = Xcolor

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked


def pad2same_size(X_list):
    _, max_x, max_y, nclasses = np.max(np.stack([v.shape for v in X_list]), 0)

    X_listpadded = [np.pad(v, [(0, 0), (0, max_x - v.shape[1]), (0, max_y - v.shape[2]), (0, 0)],
                           mode='constant', constant_values=0)
                    for v in X_list]

    return np.concatenate(X_listpadded, 0)
