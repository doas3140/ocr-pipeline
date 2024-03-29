# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/01_corner_detection_training.ipynb (unless otherwise specified).

__all__ = ['textify', 'gaussian_blur', 'get_mask_fn', 'resize_tuple', 'input_shape', 'acc_camvid', 'resize',
           'detect_object']

# Cell
from fastai import *
from fastai.vision import *
import pandas as pd
import numpy as np
import cv2
from tqdm.notebook import tqdm

# Cell
import string
from PIL import Image, ImageDraw, ImageFont

def _textify(x, p=0.9):
    val = np.random.random_sample()

    if np.random.random_sample() < p:
        pil_img = PIL.Image.fromarray(image2np(x*255).astype(np.uint8))

        w, h = pil_img.size
        text_loc = (random.randint(0,w//2),random.randint(0,h//2))
        text_color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        text_length = random.randint(1,10)
        create_string = lambda: random.sample(string.printable, 1)[0]+random.sample(string.ascii_letters, 1)[0]
        text = ''.join([create_string() for i in range(text_length)])
        text_size = random.randint(3,30)
        font = ImageFont.FreeTypeFont("../fonts/arial.ttf", size=text_size)
        ImageDraw.Draw(pil_img).text(text_loc, text, fill=text_color, font=font)

        x = pil2tensor(pil_img,np.float32)
        x.div_(255)
    return x

def textify(p=1.0):
    return RandTransform(tfm=TfmPixel(_textify), kwargs={}, p=p, resolved={}, do_run=True, is_random=True, use_on_y=False)

# Cell
def _gaussian_blur(x, size:uniform_int):
    blurred = cv2.blur(image2np(x), (size,size)) # np.arr
#     blurred = cv2.GaussianBlur(image2np(x), (size,size), 0)
    return tensor(blurred).permute(2,0,1)

# gaussian_blur = TfmPixel(_gaussian_blur)
def gaussian_blur(size, p=1.0):
    return RandTransform(tfm=TfmPixel(_gaussian_blur), kwargs={'size':size}, p=p, resolved={}, do_run=True, is_random=True, use_on_y=False)

# Cell
def get_mask_fn(filepath):
    fp_split = filepath.split('/')
    return os.path.join(*(fp_split[:3] + ['masks'] + fp_split[-3:]))[:-4] + 'png'

# Cell
resize_tuple = lambda shape, k: (int(shape[0]/k), int(shape[1]/k))
input_shape = resize_tuple([1080,1920], 10)

# Cell
def acc_camvid(input, target, void_code=0):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

# Cell
def resize(im, new_height):
    height, width = im.shape[:2]
    h_ratio = height / new_height
    im = cv2.resize(im, (int(width/h_ratio), int(height/h_ratio)))
    return im, h_ratio

def detect_object(img, image_height=100, canny_threshold=0):
    # img - (200,300) shape

    # Reduce the image
    if image_height is not None:
        img, h_ratio = resize(img, image_height)

    # Convert to grayscale if we have a color image
    if len(img.shape) == 3:
        # we keep only the luminance
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[..., 1]

    height, width = img.shape
    cx = width // 2
    cy = height // 2

    # Default value for results: return fhe whole frame if we cannot find any object
    tl_x, tl_y, bl_x, bl_y, br_x, br_y, tr_x, tr_y = ( 0., 0., 0., 0., 0., 0., 0., 0.)

    # Remove small components
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=5)
    blurred = cv2.blur(closed, (3,3))

    # Binarize
    binary = cv2.Canny(blurred.astype(np.uint8), 0, canny_threshold, 3)

    # Contour extraction
    # Since opencv 3.2 source image is not modified by this function (and the API changed...)
#     _image, contours, _hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Select the best contour
    if len(contours) > 0:
        good_contours = []  # list of contours
        distances = None  # list of floats

        # First check for contours around the center point
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            dist = cv2.pointPolygonTest(hull,(cx,cy), False)  # fast inclusion test
            if dist > 0:  # The center point is included in the convex hull of the contour
                good_contours.append(hull)

        # If no match, pick the contours with the greatest curve length
        if len(good_contours) == 0:
            dists1 = [cv2.arcLength(cnt, False) for cnt in contours]
            idx1 = np.argsort(dists1)[::-1]  # reverse order
            # Keep at most 5 contours
            good_contours = [contours[ii] for ii in idx1[:5]]

        # if we finally have good contours
        if len(good_contours) >= 0:
            # Compute exact distances to center
            distances =  [cv2.pointPolygonTest(cnt, (cx,cy), True) for cnt in good_contours]
            idx = np.argsort(distances)  # get the indices to sort the contours by distance
            # keep the 5 best contour
            for i in idx:
                best_contour = good_contours[i]
                # Polygonal approximation
                approx = cv2.approxPolyDP(best_contour,0.1*cv2.arcLength(best_contour,True),True)

                # check we have quadrilateral and assign corners
                if len(approx) == 4:
                    xs = [approx[0][0][0],approx[1][0][0],approx[2][0][0],approx[3][0][0]]
                    ys = [approx[0][0][1],approx[1][0][1],approx[2][0][1],approx[3][0][1]]
                    idx = np.argsort(ys)
                    top = [[xs[idx[0]],ys[idx[0]]],[xs[idx[1]],ys[idx[1]]]]
                    bottom = [[xs[idx[2]],ys[idx[2]]],[xs[idx[3]],ys[idx[3]]]]
                    if top[0][0] > top[1][0]:
                        tl_x, tl_y = top[1]
                        tr_x, tr_y = top[0]
                    else:
                        tl_x, tl_y = top[0]
                        tr_x, tr_y = top[1]
                    if bottom[0][0] > bottom[1][0]:
                        bl_x, bl_y = bottom[1]
                        br_x, br_y = bottom[0]
                    else:
                        bl_x, bl_y = bottom[0]
                        br_x, br_y = bottom[1]
                    break

    out = (tl_x, tl_y, bl_x, bl_y, br_x, br_y, tr_x, tr_y)

    if image_height is not None:
        out = tuple( np.array(out) * h_ratio )

    return out