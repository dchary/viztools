import numpy as np
import cv2, math, os, sys
import viztools as vz
from viztools.sequence import WellSequence
from tqdm.notebook import tqdm as tqdm

def trim_equal(img):
    h, w = img.shape[2:]
    if w - h > 0:
        img = img[:,:,:, 0:img.shape[2] - (w-h),]
    elif w - h < 0:
        img = img[:,:,0:img.shape[2] + (w-h),:]
    return img

def to_8bit(img):
    # Rescale each channel to unsigned 8-bit (values can be between 0 and 255)
    minmax = lambda x : (x - x.min()) / (x.max() - x.min())
    for channel in range(img.shape[1]):
        #for timestep in range(img.shape[0]):
        img[:, channel, :, :] = (minmax(img[:, channel, :, :]) * 255.0)

    # Explicit type redefinition to unsigned 8-bit
    img = np.uint8(img)
    return img

def to_8bit_one(img):
    # Rescale each channel to unsigned 8-bit (values can be between 0 and 255)
    minmax = lambda x : (x - x.min()) / (x.max() - x.min())
    img = minmax(img) * 255.0

    # Explicit type redefinition to unsigned 8-bit
    img = np.uint8(img)
    return img




def _find_nearest_above(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, -1)
    # We need to mask the negative differences
    # since we are looking for values above
    if np.all(mask):
        c = np.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()

def _hist_match(original, specified):

    oldshape = original.shape
    original = original.ravel()
    specified = specified.ravel()

    # get the set of unique pixel values and their corresponding indices and counts
    s_values, bin_idx, s_counts = np.unique(original, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(specified, return_counts=True)

    # Calculate s_k for original image
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    
    # Calculate s_k for specified image
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # Round the values
    sour = np.around(s_quantiles*255)
    temp = np.around(t_quantiles*255)
    
    # Map the rounded values
    b=[]
    for data in sour[:]:
        b.append(_find_nearest_above(temp,data))
    b= np.array(b,dtype='uint8')

    return b[bin_idx].reshape(oldshape)

def equalize_histogram(img, channels = [0]):
    # Equalize histograms
    for channel in channels:
        for timestep in range(1, img.shape[0]):
            img[timestep,channel,:,:] = _hist_match(img[timestep,channel,:,:], img[0,channel,:,:])

    # Equalize the first channel based on the last
    for channel in channels:
        img[0,channel,:,:] = _hist_match(_hist_match(img[0,channel,:,:], img[1,channel,:,:]), img[1,channel,:,:])
    return img


def autoscale_histogram(img, channels = [0,1,2,3,4]):

    minmax = lambda x : (x - x.min()) / (x.max() - x.min())

    for channel in channels:
        img_channel = img[:,channel, 0, 0]

        channel_mean = img_channel.mean()
        channel_std = img_channel.std()
        
        channel_min = np.maximum(0, channel_mean - (3*channel_std))
        channel_max = np.minimum(255, channel_mean + (3*channel_std))

        img_channel_new = ((channel_max - channel_min) * minmax(img_channel)) + channel_min

        img[:, channel, :, :] = img_channel_new


def apply_deskew_transform(img, s, M):
    for timestep in range(img.shape[0]):
        for channel in range(img.shape[1]):
            img[timestep, channel, :, :] = cv2.warpAffine(img[timestep, channel, :, :], M, s, borderMode=cv2.BORDER_REPLICATE)
    return img

def deskew_image(img):

    # Get a median frame off the briightfield channel
    frame = np.median(img[0:1,0,:,:], axis = 0).astype(np.uint8)

    height, width = frame.shape

    # Denoise it
    frame = cv2.fastNlMeansDenoising(frame, h=3)

    # Create an inverted B&W copy using Otsu (automatic) thresholding
    frame = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Detect edges
    lines = cv2.HoughLinesP(frame, 1, np.pi / 180, 200, minLineLength=width / 12, maxLineGap=width / 150)

    # Collect the angles of these lines (in radians)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angles.append(np.arctan2(y2 - y1, x2 - x1))

    # If the majority of our lines are vertical, this is probably a landscape image
    landscape = np.sum([abs(angle) > np.pi / 4 for angle in angles]) > len(angles) / 2

    # Filter the angles to remove outliers based on max_skew
    max_skew=45

    if landscape:
        angles = [angle for angle in angles if np.deg2rad(90 - max_skew) < abs(angle) < np.deg2rad(90 + max_skew)]
    else:
        angles = [angle for angle in angles if abs(angle) < np.deg2rad(max_skew)]

    # Average the angles to a degree offset
    angle_deg = np.rad2deg(np.median(angles))

    # Rotate the image by the residual offset
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1)

    img = apply_deskew_transform(img, (width, height), M)
    img = np.uint8(img)

    return img

def apply_transform(f0, M):
    return cv2.warpAffine(f0, M, dsize = f0.shape)

def find_homography(f0, f1):

    # Detect feature points in previous frame
    prev_pts = cv2.goodFeaturesToTrack(f0, maxCorners=5000, qualityLevel=0.1, minDistance=5, blockSize=3)
    
    # Calculate optical flow (i.e. track feature points)
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(f0, f1, prev_pts, None)

    # Sanity check that both point lists are equal
    assert prev_pts.shape == curr_pts.shape

    # Filter only valid points
    idx = np.where(status==1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    # Estimate transformation matrix between frames
    transformation_rigid_matrix, rigid_mask = cv2.estimateAffinePartial2D(curr_pts, prev_pts)

    #f1 = cv2.warpAffine(f1, transformation_rigid_matrix, dsize = f1.shape, borderMode=cv2.BORDER_REPLICATE)
    return transformation_rigid_matrix

def stabilize_image(img):
    
    # Stabilize motion between frames
    for timepoint in tqdm(range(1, img.shape[0]), leave = False):

        # Find a transformation that will match the two brightfield frames 0 and 1
        # Find a transformation that will match the intersection of the non-brightfield frames
        f0 = img[timepoint - 1, 1:,:,:].sum(0).astype(np.uint8)
        f1 = img[timepoint, 1:,:,:].sum(0).astype(np.uint8)
        M = find_homography(f0, f1)

        # Apply transformation to each channel individually
        for channel in range(img.shape[1]):
            img[timepoint, channel, :, :] = apply_transform(img[timepoint, channel, :, :], M) 
    
    return img

def auto_scale_channels(img, smin = 3, smax = 10):
    for channel in range(img.shape[1] - 1):
        for timestep in range(1, img.shape[0]):
            img[timestep,channel,:,:] = _auto_scale(img[timestep,channel,:,:], smin = smin, smax = smax)

    return img

def _auto_scale(img, smin = 3, smax = 10):

    baseline = int(img.mean())
    min_val = int(min(255, max(0, baseline - (smin * img.std()))))
    max_val = int(min(255, max(0, baseline + (smax * img.std()))))

    return cv2.threshold(img, min_val, max_val, cv2.THRESH_TOZERO)[1]

def _auto_scale_otsu(img, thresh):

    # Apply manual thresholding
    _, thresh_image = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

    # Define a kernel for morphological operations
    kernel = np.ones((3,3), np.uint8)

    # Apply opening to remove noise
    cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel)

    return cleaned_image

def auto_scale_otsu(img, thresh):
    for channel in range(img.shape[1] - 1):
        for timestep in range(1, img.shape[0]):
            img[timestep,channel,:,:] = _auto_scale_otsu(img[timestep,channel,:,:], thresh = thresh)
    return img

def recipe_default(ws : WellSequence):

    X = ws.get_X()

    X = trim_equal(X)
    X = to_8bit(X)
    X = equalize_histogram(X)
    X = deskew_image(X)
    X = trim_equal(X)
    X = stabilize_image(X)
    X = auto_scale_otsu(X, 45)

    ws.set_X(X)

    return ws
