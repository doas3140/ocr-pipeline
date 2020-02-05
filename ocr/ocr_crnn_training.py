# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/05_ocr_crnn_training.ipynb (unless otherwise specified).

__all__ = ['PAD', 'PAD', 'DATA_PATH', 'crnn_config', 'allowed_chars', 'allowed_chars', 'MultiCategoryProcessor',
           'MultiCategory', 'one_hot_text', 'MultiCategoryList', 'im2seq_data_collate', 'str2lines', 'MyImageList',
           'gaussian_blur', 'resize_tfm', 'rand_resize', 'resize_one_img', 'train_transforms', 'valid_transforms',
           'create_data', 'conv_output', 'CRNN', 'image_width2seq_len', 'loss_func', 'print_metric', 'decode_ctc',
           'wer', 'word_error', 'char_error', 'CER', 'WER']

# Cell
from fastai import *
from fastai.vision import *
import pandas as pd
import numpy as np
import cv2
from tqdm.notebook import tqdm

# Cell
from .core import save_inference, load_inference
from .ocr_dataset_fontsynth import create_df as create_fontsynth_df
from .ocr_dataset_sroie2019 import create_df as create_sroie_df
from .ocr_dataset_sroie2019 import sroie_ocr_config, DATA_PATH, char_freq
from .ocr_dataset_fontsynth import fontsynth_config, char_freq
PAD = sroie_ocr_config.PAD # PAD - how much is data padded
PAD = 0
DATA_PATH = fontsynth_config.LINES_DIR

# Cell
allowed_chars = {'N', '3', 'V', 'P', '7', '1', '#', '9', '"', 'C', 'Q', 'B', 'E', '>', '@', ',', 'M', '{', ']',
                 ';', '^', "'", '&', '6', 'Z', '*', '<', '+', 'G', 'X', '!', ':', '-', '[', '|', '$', '5', 'I',
                 'H', '=', 'Y', '.', 'R', 'S', '/', 'T', '}', 'K', '0', '?', 'U', ')', '_', 'D', 'J', 'L', '4',
                 'W', '%', '(', '\n', ' ', 'F', '8', '~', '\\', 'A', '2', 'O'}

allowed_chars = fontsynth_config.allowed_chars

class crnn_config:
    LINE_HEIGHT = 48
    USE_DEFAULT_CLASSES = True
    label_delim = '`'
    allowed_chars = allowed_chars

# Cell
# label_delim = '`' # '<pad>''

class MultiCategoryProcessor(PreProcessor):
    "`PreProcessor` that create `classes` from `ds.items` and handle the mapping."
    def __init__(self, ds:ItemList):
        self.create_classes(ds.classes)
        self.use_default_classes = crnn_config.USE_DEFAULT_CLASSES
        self.default_classes = crnn_config.allowed_chars

    def create_classes(self, classes):
        self.classes = classes
        if classes is not None: self.c2i = {v:k for k,v in enumerate(classes)}

    def process_one(self,item):
        ''' list of chars from `MultiCategoryList.get()` '''
        return [ self.c2i[c] if c in self.c2i else 0 for c in item ]

    def process(self, ds):
        if self.classes is None: self.create_classes(self.generate_classes(ds.items))
        ds.classes = self.classes
        ds.c2i = self.c2i
        super().process(ds)

    def generate_classes(self, items):
        ''' items = [ ['h', 'e', 'l', 'l', 'o'], [...], ...] '''
        "Generate classes from `items` by taking the sorted unique values."
        if self.use_default_classes:
            classes = list(self.default_classes)
        else:
            classes = set()
            for c in items: classes = classes.union(set(c))
            classes = list(classes)
        classes.sort()
        return [crnn_config.label_delim] + classes # CHANGED

# Cell
class MultiCategory(ItemBase):
    "Basic class for multi-classification labels."
    def __init__(self, data, obj, raw): self.data, self.obj, self.raw = data, obj, raw
    def __str__(self):  return crnn_config.label_delim.join([str(o) for o in self.obj])
    def __hash__(self): return hash(str(self))

# Cell
def one_hot_text(x:Collection[int], c:int):
    "One-hot encode `x` with `c` classes."
    ''' x w/ len of n returns [n,c] shape arr '''
    res = np.zeros((len(x),c), np.float32)
    res[np.arange(len(x)), listify(x)] = 1.
    return res

# Cell
class MultiCategoryList(ItemList):
    "Basic `ItemList` for multi-classification labels."
    _processor = MultiCategoryProcessor
    def __init__(self, items:Iterator, classes:Collection=None, label_delim:str=None, one_hot:bool=False, **kwargs):
        self.classes = classes
        items = [line.split(crnn_config.label_delim) for line in items] # CHANGED
        super().__init__(items, **kwargs)
        self.processor = [MultiCategoryProcessor(self)]

    def get(self, i):
        o = self.items[i] # list of ints that represent chars
        return MultiCategory(tensor(o), [self.classes[p] for p in o], o) # CHANGED

    def analyze_pred(self, pred, thresh:float=0.5):
        return (pred >= thresh).float()

    def reconstruct(self, data_out):
        if isinstance(data_out, list): # output of data
            t_argmax, _, lengths = data_out
        else: # output from nn
            t_argmax = torch.argmax(data_out, axis=-1) # CHANGED
#         t = data_out[0] if isinstance(data_out, list) else data_out # if train mode it returns tuple
        ''' t [n,c] tensor '''
        o = [int(i) for i in t_argmax] # CHANGED
        return MultiCategory(one_hot_text(o, self.c), [self.classes[p] for p in o], o)

    @property
    def c(self): return len(self.classes)

# Cell
def im2seq_data_collate(batch:ItemsList, pad_idx:int=0)->Tensor:
    if isinstance(batch[0][1], int): return data_collate(batch)
    "Convert `batch` items to tensor data."
    data = to_data(batch) # list of (image, text) pairs
    # image: [3,48,w], text: [n,c], where n's and w's are different
    max_w = max([image.shape[2] for image, text in data])
    max_h = max([image.shape[1] for image, text in data])
    max_n = max([text.shape[0] for image, text in data])
#     _, num_classes = data[0][1].shape

    images = torch.zeros(len(batch), 3, max_h, max_w)
#     texts = torch.zeros(len(batch), max_n, num_classes)
    texts = []
    nn_out_seq_len, texts_len = [], []
    for i, (image, text) in enumerate(data):
        c,h,w = image.shape
        images[i, : , : , :w ] = image
        images[i, : , : , w: ] = image[:,:,w-1].unsqueeze(2).expand(c,h,max_w-w)
        nn_out_seq_len.append( image_width2seq_len(w) )
        n = text.size(0)
        texts.append( tensor(text) )
#         texts[i, :n , : ] = tensor(text)
#         texts[i, n: , -1 ] = 1
        texts_len.append(n)
#     texts = torch.cat(texts, axis=0)
    return images, (texts, tensor(nn_out_seq_len).type(torch.int), tensor(texts_len).type(torch.int))

# Cell
def str2lines(string, n=50):
    return ''.join([s+'\n' if (i+1)%n == 0 else s for i,s in enumerate(string)])

str2lines('asdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasdasd')

# Cell
class MyImageList(ImageList):
    def show_xys(self, xs, ys, imgsize:int=4, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show the `xs` (inputs) and `ys` (targets) on a figure of `figsize`."
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, 1, imgsize=imgsize, figsize=figsize) # CHANGED rows -> 1
        for x,y,ax in zip(xs, ys, axs.flatten()): x.show(ax=ax, y=y, **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
        plt.tight_layout()

    def show_xyzs(self, xs, ys, zs, imgsize:int=20, figsize:Optional[Tuple[int,int]]=None, **kwargs):
        "Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`."
#         if self._square_show_res:
        title = 'Ground truth\nPredictions'
        rows = int(np.ceil(math.sqrt(len(xs))))
        axs = subplots(rows, 1, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=12) # CHANGED rows -> 1
        for x,y,z,ax in zip(xs,ys,zs,axs.flatten()):
            x.show(ax=ax, title=f'y_true: {str2lines(str(y))}\n\ny_pred: {str2lines(str(z))}', **kwargs)
        for ax in axs.flatten()[len(xs):]: ax.axis('off')
#         else:
#             title = 'Ground truth/Predictions'
#             axs = subplots(len(xs), 2, imgsize=imgsize, figsize=figsize, title=title, weight='bold', size=14)
#             for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
#                 x.show(ax=axs[i,0], y=y, **kwargs)
#                 x.show(ax=axs[i,1], y=z, **kwargs)

# Cell
def _gaussian_blur(x, size:uniform_int):
    blurred = cv2.blur(image2np(x), (size,size)) # np.arr
#     blurred = cv2.GaussianBlur(image2np(x), (size,size), 0)
    return tensor(blurred).permute(2,0,1)

def gaussian_blur(size, p=1.0):
    return RandTransform(tfm=TfmPixel(_gaussian_blur), kwargs={'size':size}, p=p, resolved={}, do_run=True, is_random=True, use_on_y=False)

# Cell
resize_one_img = lambda x, size: F.interpolate(x[None], size=size, mode='bilinear', align_corners=True)[0]

def resize_tfm(x, pad:uniform_int, line_height=crnn_config.LINE_HEIGHT):
    ''' size of subtracted padding '''
    c,h,w = x.shape
    x = x[ : , pad:h-pad , pad:w-pad ]
    new_w = int(w * line_height / float(h))
    return resize_one_img(x, size=(line_height, new_w))

def rand_resize(pad, p=1.0):
    return RandTransform(tfm=TfmPixel(resize_tfm), kwargs={'pad':pad}, p=p, resolved={}, do_run=True, is_random=True, use_on_y=False)

# Cell
train_transforms = [
    rand_resize(pad=(0,PAD), p=1.0),
    rotate(degrees=(-1, 1), p=0.6),
    symmetric_warp(magnitude=(-0.03, 0.03), p=0.1),
    rand_zoom(scale=(0.9,1.03), p=0.5),
    brightness(change=(0.35, 0.65), p=0.4),
    contrast(scale=(0.7,1.3), p=0.4),
    gaussian_blur(size=(1, 7), p=0.2),
#     squish(scale=(0.85,1.15), p=0.3),
#     cutout(n_holes=(0,6), length=(1,10)), # black rect
#     tilt(direction=(0,3), magnitude=(-0.2,0.2), p=0.3)
]

valid_transforms = [
    rand_resize(pad=(0,0), p=1.0) # (no padding, but need to resize)
]

# Cell
def create_data(df, bs=32):
    ''' DataFrame (df) -> Dataloader (dl) '''
    data = (MyImageList.from_df(df, path=DATA_PATH, cols='image_path')
        .split_from_df(col='valid')
        .label_from_df(cols='string', label_cls=MultiCategoryList, label_delim=crnn_config.label_delim)
        .transform((train_transforms, valid_transforms), tfm_y=False)
        .databunch(bs=bs, collate_fn=im2seq_data_collate)
        .normalize(imagenet_stats)
    )
    return data

# Cell
def conv_output(w, ss, ps=None, ks=3):
    ''' image width, strides, pools, kernel sizes '''
    for s,p,k in zip(ss,ps,ks):
        s = s[1] if isinstance(s, tuple) else s
        w = w if w%s == 0 else w + 1
        w = (w - k + 2*p)/s + 1 if p is not None else w/s
    return int(w)

# Cell
def _create_cnn(kernels, strides, channels, padding):
    layers = []
    for i,o,k,s,p in zip([3] + channels[:-1], channels, kernels, strides, padding):
        layers.append( conv_layer(ni=i, nf=o, ks=k, stride=s, padding=p) )
    return nn.Sequential(*layers)

# Cell
class CRNN(nn.Module):

    def __init__(self, nclass=10, nc=3, rnn_hidden=256, bidirectional=True):
        super(CRNN, self).__init__()
        self.nclass = nclass
        kernels = [3, 3, 3, 3, 3]
        strides = [2, 2, (2,1), (2,1), 1]
        channels = [64, 128, 256, 256, 512]
        padding = [None] * 4 + [0] # None - out size doesnt change

        self.kernels, self.strides, self.channels, self.padding = kernels, strides, channels, padding

        self.cnn = _create_cnn(kernels, strides, channels, padding)

        self.rnn = nn.LSTM(channels[-1], rnn_hidden, bidirectional=bidirectional)
        mult = 1 if not bidirectional else 2
        self.linear = nn.Linear(rnn_hidden * mult, nclass)

#     def eval(self): # (quick fix) model.eval() returns bad outputs w/ BatchNorm
#         return self

    def forward(self, x): # input: [b, 1, h, w]
        # output: ([b,s,c], [b]) (output, seq lengths)
        x = self.cnn(x) # [b,512,1,w/4-2]
        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        # [b,512,1,w] -> [s,b,512] (w == s)
        x = x.squeeze(2).permute(2, 0, 1) # [b,c,w] (w == s)
        # [s,b,512] -> [s,b,c]
        x, _ = self.rnn(x)
        x = self.linear(x)
#         nn_output = (x, pad_mask)
        return x.permute(1,0,2)

# Cell
image_width2seq_len = lambda w: conv_output(w, crnn.strides, crnn.padding, crnn.kernels)

# Cell
def loss_func(y_pred, y_true, y_pred_len, y_true_len):
    # y_pred: [b,s_e,c], y_true: [[s_d], [s_d], ...], lengths: [b]
    b, s_e, c = y_pred.shape
    y_true = torch.cat(y_true, axis=0) # [b*s_d]
    y_pred = y_pred.log_softmax(axis=2).permute(1,0,2) # [ s_e, b, c ]
    torch.backends.cudnn.enabled = False
    loss = ctc_loss(y_pred, y_true, y_pred_len, y_true_len)
    torch.backends.cudnn.enabled = True
    return loss

# Cell
def print_metric(y_pred, y_true):
#     o = y_pred.argmax(-1)[0].cpu().numpy()
#     print('PRED:', MultiCategory(one_hot_text(o, data.c), [data.classes[p] for p in o], o))
#     o = y_true.argmax(-1)[0].cpu().numpy()
#     print('TRUE:', MultiCategory(one_hot_text(o, data.c), [data.classes[p] for p in o], o))
    return tensor(0)

# Cell
def decode_ctc(texts, classes):
    """ convert text-index into text-label. (make sure len(t) doesnt throw length of 0-dim error) """
    out = []
    index = 0
    for t in texts:
        char_list = []
        for i in range(len(t)):
            if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                char_list.append(t[i])
#         text = ''.join(char_list)

        out.append(tensor(char_list))
    return out

# Cell
def wer(s1,s2):
    ''' s1 - true text, s2 - pred text '''
    d = np.zeros([len(s1)+1,len(s2)+1])
    d[:,0] = np.arange(len(s1)+1)
    d[0,:] = np.arange(len(s2)+1)

    for j in range(1,len(s2)+1):
        for i in range(1,len(s1)+1):
            if s1[i-1] == s2[j-1]:
                d[i,j] = d[i-1,j-1]
            else:
                d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+1)

    return d[-1,-1]/len(s1)

word_error = wer( 'black frog jumped away'.split(' '), 'black frog jumped awayyy'.split(' ') )
char_error = wer( 'black frog jumped away', 'black frog jumped awayyy' )
char_error, word_error

# Cell
def CER(y_pred, y_true, y_pred_len, y_true_len):
    # y_pred: [b,s_e,c], y_true: [[s_d], [s_d], ...], lengths: [b]
    y_pred = y_pred.argmax(-1)
    y_pred = decode(y_pred)
    m = 0
    for yp, p_len, yt in zip(y_pred, y_pred_len, y_true):
        if yp.shape == torch.Size([]): continue
        yt_text = ''.join([learner.data.classes[i] for i in yt])
        yp_text = ''.join([learner.data.classes[i] for i in yp])
        m += wer(yt_text, yp_text)
    return tensor(m / len(y_pred))

# Cell
def WER(y_pred, y_true, y_pred_len, y_true_len):
    # y_pred: [b,s_e,c], y_true: [[s_d], [s_d], ...], lengths: [b]
    y_pred = y_pred.argmax(-1)
    y_pred = decode(y_pred)
    m = 0
    for yp, p_len, yt in zip(y_pred, y_pred_len, y_true):
        if yp.shape == torch.Size([]): continue
        yt_text = ''.join([learner.data.classes[i] for i in yt])
        yp_text = ''.join([learner.data.classes[i] for i in yp])
        m += wer(yt_text.split(' '), yp_text.split(' '))
    return tensor(m / len(y_pred))