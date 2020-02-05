from flask import Flask, request, Response, send_file
import jsonpickle
import numpy as np
import os
from tqdm import tqdm
import cv2
from time import time

from fastai import *
from fastai.vision import *

app = Flask(__name__)


### MODELS ###


from ocr.dewarp import dewarp_image

from ocr.corner_detection_training import *
from ocr.corner_detection_inference import PaperCornerPredictor, points_on_mask
corner_predictor_unet = PaperCornerPredictor(model_name='unet_paper_mask', use_unet=True, use_gpu=False)
corner_predictor_cv = PaperCornerPredictor(model_name='unet_paper_mask', use_unet=False, use_gpu=False)

def image2corner_points(img, config):
    predictor = corner_predictor_unet if config['corner_predictor'] == 'unet' else corner_predictor_cv
    pts_tensor = predictor.predict_corners(tensor(img), shape='train')
    return [int(i) for i in pts_tensor]

from ocr.bbox_east_training import *
from ocr.bbox_east_inference import TextBBoxPredictor, bboxes_on_image
text_predictor = TextBBoxPredictor(model_name='east_bbox_10x3', use_gpu=False)

def image2bboxes(img, config):
    bboxes_tensor = text_predictor.image2bboxes(tensor(img)) # [N,4]
    return [[int(i) for i in bb] for bb in bboxes_tensor]

from ocr.ocr_crnn_training import *
from ocr.ocr_inference import CrnnOcrPredictor
crnn_predictor = CrnnOcrPredictor(model_name='crnn_ocr', use_gpu=False)

from ocr.ocr_inference import TesseractOcrPredictor
tesseract_predictor = TesseractOcrPredictor()

from ocr.ocr_attention_training import *
from ocr.ocr_inference import AttentionOcrPredictor
attention_predictor = AttentionOcrPredictor(model_name='attention_ocr', use_gpu=False)

MODELS = {
    'crnn': crnn_predictor,
    'tesseract': tesseract_predictor,
    'attention': attention_predictor,
}

def image2text(img, config):
    ocr_predictor = MODELS[config['model']] if config['model'] is not None else MODELS['attention']
    return ocr_predictor.image2text(tensor(img))


### OCR MAIN ###


class PT(object): # Print Times
    def __init__(self, name):
        print(f'{name}... ', end='')
    def __enter__(self): 
        self.t0 = time()
    def __exit__(self, exc_type, exc_value, tb): 
        print(f'took: {time()-self.t0}')


def image2ocr(img, config):
    if config['corner_predictor'] != 'none':
        with PT('detecting corners') as _: pts = image2corner_points(img, config)
        with PT('transforming image') as _: img = corner_predictor_cv.transform_image(img, pts)

    if config['dewarp'] == 'true':
        with PT('dewarping image') as _: img = dewarp_image(img)

    with PT('finding text') as _: bboxes = image2bboxes(img, config) # [N,4]

    out = [] # list( {'location': [4], 'text': str}, ... )
    with PT('transforming image') as _:
        for t,l,b,r in tqdm(bboxes):
            text_im = img[ t:b , l:r ]
            text = image2text(text_im, config)
            out.append({'location': [t,l,b,r], 'text': text})

    return out


### ROUTES ###


def decode_image(img_bytes):
    img_raw = np.frombuffer(img_bytes, np.uint8) # <class 'bytes'> -> np.arr
    img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
    img = img[ : , : , [2,1,0] ] # BGR -> RGB
    return img

def send_image(img, config):
    r = request.args.get('resize')
    with tempfile.NamedTemporaryFile() as f:
        img = img[ : , : , [2,1,0] ] # RGB -> BGR
        if r is not None: img = cv2.resize(img, dsize=None, fx=float(r), fy=float(r))
        cv2.imwrite(f'{f.name}.jpg', img)
        return send_file(f'{f.name}.jpg', mimetype='image/gif') # as_attachment=True


''' @input:
@input: Body = {'image': File})
@output: {'message': 'image received...'}
'''
@app.route('/api/test', methods=['POST'])
def test():
    img = decode_image(img_bytes=request.files['image'].read())
    response = {'message': 'image received. img.shape: {}x{}'.format(img.shape[1], img.shape[0])}
    return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")


''' 
@input: (
    Body = {'image': File},
    Params = {
        'resize': '0.5' (how much to resize output image)
    }
)

@output: jpg image
'''
@app.route('/api/dewarp_v1', methods=['POST'])
def dewarp_v1():
    config = dict(request.args)
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        img = dewarp_image(img)
        return send_image(img, config=config)
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


''' 
@input: (
    Body = {'image': File},
    Params = {
        'resize': '0.5', (how much to resize output image)
        'corner_predictor': 'unet'/'cv'/'none', (default is 'cv')
    }
)

@output: jpg image
'''
@app.route('/api/predict_corners_v1', methods=['POST'])
def predict_corners_v1():
    config = dict(request.args)
    if 'corner_predictor' not in config: config['corner_predictor'] = 'cv'
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        pts = image2corner_points(img, config=config)
        img = points_on_mask(img, pts)
        return send_image(img, config=config)
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


''' 
@input: (
    Body = {'image': File},
    Params = {
        'resize': '0.5', (how much to resize output image)
    }
)

@output: jpg image
'''
@app.route('/api/predict_bboxes_v1', methods=['POST'])
def predict_bboxes_v1():
    config = dict(request.args)
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        bboxes = image2bboxes(img, config=config)
        img = bboxes_on_image(img, bboxes)
        return send_image(img, config=config)
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


''' @input:
@input: (
    Body = {'image': File},
    Params = {
        'model': 'crnn'/'tesseract'/'attention'/..., (select one from MODELS dict above) (default is 'tesseract')
        'corner_predictor': 'unet'/'cv'/'none', (default is 'none')
        'dewarp': 'true'/'false', (default is 'false')
        'font_scale': '1.0', (how big is text) (default is '1.0')
    }
) (if Params are not specified, default config is used [look at parse_config func])

@output: jpg image
'''
@app.route('/api/ocr_v1_preview', methods=['POST'])
def ocr_v1_preview():
    import cv2

    config = dict(request.args)
    if 'corner_predictor' not in config: config['corner_predictor'] = 'none'
    if 'dewarp' not in config: config['dewarp'] = 'false'
    if 'model' not in config: config['model'] = 'attention'
    if 'font_scale' not in config: config['font_scale'] = '1.0'
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        ocr_list = image2ocr(img, config=config)
        img = np.ascontiguousarray(img) # WTF: https://github.com/opencv/opencv/issues/14866
        for element in ocr_list:
            t,l,b,r = element['location']
            text = element['text']
            img = cv2.rectangle(img, (l,t), (r,b), color=(255,255,0), thickness=2)
            img = cv2.putText(img, text, org=(l+10,b-10), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                                fontScale=2.0, color=(255,255,0), thickness=5, lineType=cv2.LINE_AA) 
        return send_image(img, config=config)
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


''' @input:
@input: (
    Body = {'image': File},
    Params = {
        'model': 'crnn'/'tesseract'/'attention'/..., (select one from MODELS dict above) (default is 'tesseract')
        'corner_predictor': 'unet'/'cv'/'none', (default is 'none')
        'dewarp': 'true'/'false', (default is 'false')
        'font_scale': '1.0', (how big is text) (default is '1.0')
    }
) (if Params are not specified, default config is used [look at parse_config func])

@output: {
    'ocr': [
        {'location': int[4]{t,l,b,r}, 'text': str},
        ...
    ]
}
'''
@app.route('/api/ocr_v1', methods=['POST'])
def ocr_v1():
    import cv2

    config = dict(request.args)
    if 'corner_predictor' not in config: config['corner_predictor'] = 'none'
    if 'dewarp' not in config: config['dewarp'] = 'false'
    if 'model' not in config: config['model'] = 'attention'
    if 'font_scale' not in config: config['font_scale'] = '1.0'
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        ocr_list = image2ocr(img, config=config)
        return Response(response=jsonpickle.encode({'ocr': ocr_list}), status=200, mimetype="application/json")
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


    return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")




app.run(host="0.0.0.0", port=5000)