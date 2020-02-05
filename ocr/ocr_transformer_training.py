# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/08_ocr_transformer_training.ipynb (unless otherwise specified).

__all__ = ['PAD', 'PAD', 'DATA_PATH', 'transformer_config', 'allowed_chars', 'allowed_chars', 'MultiCategoryProcessor',
           'label_delim', 'eos', 'bos', 'pad_idx', 'bos_idx', 'eos_idx', 'MultiCategory', 'one_hot_text',
           'MultiCategoryList', 'str2lines', 'MyImageList', 'gaussian_blur', 'resize_tfm', 'rand_resize',
           'resize_one_img', 'train_transforms', 'valid_transforms', 'create_data', 'conv_output', 'MultiHeadAttention',
           'MultiCnnLayer', 'CNN', 'TransformerModel', 'wer', 'word_error', 'char_error', 'CER', 'WER']

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
allowed_chars = {'L', '*', ':', ' ', 'C', '.', 'D', '%', '\n', '-', '"', 'J', '[', ']', 'H', '1', '<', '@',
                 'W', 'K', '+', 'Y', '7', '?', 'T', '5', '!', '#', 'P', '&', 'U', '$', 'G', ';', '~', "'",
                 ')', 'V', '_', 'O', ',', '/', 'Q', '0', '4', 'B', '=', '9', '8', '3', '>', '6', 'Z', '\\',
                 'F', 'X', 'R', 'I', 'E', 'S', '|', '{', '^', 'A', '}', '2', 'M', 'N', '('}

allowed_chars = fontsynth_config.allowed_chars

class transformer_config:
    LINE_HEIGHT = 48
    USE_DEFAULT_CLASSES = True
    eos = '</s>'
    bos = '<s>'
    label_delim = '`'
    pad_idx = 0
    bos_idx = 1
    eos_idx = 2
    allowed_chars = allowed_chars

# Cell
label_delim = transformer_config.label_delim
eos = transformer_config.eos
bos = transformer_config.bos

pad_idx = transformer_config.pad_idx
bos_idx = transformer_config.bos_idx
eos_idx = transformer_config.eos_idx

class MultiCategoryProcessor(PreProcessor):
    "`PreProcessor` that create `classes` from `ds.items` and handle the mapping."
    def __init__(self, ds:ItemList):
        self.create_classes(ds.classes)
        self.use_default_classes = transformer_config.USE_DEFAULT_CLASSES
        self.default_classes = transformer_config.allowed_chars

    def create_classes(self, classes):
        self.classes = classes
        if classes is not None: self.c2i = {v:k for k,v in enumerate(classes)}

    def process_one(self,item):
        ''' list of chars from `MultiCategoryList.get()` '''
        return [bos_idx] + [ self.c2i[c] if c in self.c2i else 0 for c in item ] + [eos_idx]

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
        return [label_delim, bos, eos] + classes # CHANGED

# Cell
class MultiCategory(ItemBase):
    "Basic class for multi-classification labels."
    def __init__(self, data, obj, raw): self.data, self.obj, self.raw = data, obj, raw
    def __str__(self):  return label_delim.join([str(o) for o in self.obj])
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
        items = [line.split(label_delim) for line in items] # CHANGED
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
#             t_argmax = torch.argmax(data_out, axis=-1) # CHANGED
            t_argmax = data_out # CHANGED
#         t = data_out[0] if isinstance(data_out, list) else data_out # if train mode it returns tuple
        ''' t [n,c] tensor '''
        o = [int(i) for i in t_argmax] # CHANGED
        return MultiCategory(one_hot_text(o, self.c), [self.classes[p] for p in o], o)

    @property
    def c(self): return len(self.classes)

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

def resize_tfm(x, pad:uniform_int, line_height=transformer_config.LINE_HEIGHT):
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
        .label_from_df(cols='string', label_cls=MultiCategoryList, label_delim=label_delim)
        .transform((train_transforms, valid_transforms), tfm_y=False)
        .databunch(bs=bs, collate_fn=im2seq_data_collate)
        .normalize(imagenet_stats)
    )
    data.train_dl.numworkers=0
    data.valid_dl.numworkers=0

#     def add_beggining_and_end(b):
#         x,y = b
#         y = F.pad(y, (1, 0), value=bos_idx)
#         y = F.pad(y, (0, 1), value=eos_idx)
#         return x,y

#     data.add_tfm(add_beggining_and_end)
    return data

# Cell
def conv_output(w, ss, ps=None, ks=3):
    ''' image width, strides, pools, kernel sizes '''
    for s,p,k in zip(ss,ps,ks):
        s = s[1] if isinstance(s, tuple) else s
        w = w if w%s == 0 else w + 1
        w = (w - k + 2*p)/s + 1 if p is not None else w/s
    return int(w)

conv_output(129, [2, 1, 2, 1, (2,1), (2,1), 1], [None] * 6 + [0], [3, 3, 3, 3, 3, 3, 3])

# Cell
_apply_layer = lambda args: args[1](args[0]) # args[0]: x, args[1]: layer => layer(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head=None, p=0., bias=True, scale=True, shared_qk=False):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        self.q_wgt, self.v_wgt = [nn.Linear(d_model, n_heads*d_head, bias=bias) for o in range(2)]
        self.k_wgt = self.q_wgt if shared_qk else nn.Linear(d_model, n_heads*d_head, bias=bias)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(p),nn.Dropout(p)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q, kv, mask=None):
        ''' [b,s_d,512], [b,s_e,512], [1,1,s_d,s_e] -> [b,s_d,512] '''
        bs,seq_len = q.size(0),q.size(1)
        wq,wk,wv = map(_apply_layer, zip([q,kv,kv], [self.q_wgt,self.k_wgt,self.v_wgt])) # [b,s_d,h*512], [b,s_e,h*512] x 2
        wq,wk,wv = map(lambda x:x.view(bs, x.size(1), self.n_heads, self.d_head), (wq,wk,wv)) # [b,s_d,h,512], [b,s_e,h,512] x 2
        wq,wv = map(lambda x:x.permute(0, 2, 1, 3), (wq,wv)) # [b,h,s_d,512], [b,h,s_e,512]
        wk = wk.permute(0, 2, 3, 1) # [b,h,512,s_e]
        attn_score = torch.matmul(wq, wk) # [b,h,s_d,s_e]
        if self.scale: attn_score.div_(self.d_head ** 0.5)
        if mask is not None: # NOTE: masks only ones, not zeros!
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score) # [b,h,s_d,s_e]
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1)) # [b,h,s_d,s_e]
        attn_vec = torch.matmul(attn_prob, wv) # [b,h,s_d,512]
        attn_vec = attn_vec.permute(0, 2, 1, 3).contiguous().contiguous() # [b,s_d,h,512]
        attention = attn_vec.view(bs, seq_len, -1) # [b,s_d,h*512]
        return self.ln(q + self.drop_res(self.out(attention)))

# Cell
class MultiCnnLayer(nn.Module):
    def __init__(self, ni, nf, ks, stride, padding):
        super().__init__()
        self.conv = conv_layer(ni=ni, nf=nf, ks=ks, stride=stride, padding=padding)
        self.pool = nn.MaxPool2d(stride)
        self.conv2 = conv_layer(ni=ni, nf=nf, ks=1, stride=1, padding=padding)

    def forward(self, x):
        return self.conv2(self.pool(x)) + self.conv(x)
#         return self.bn(self.relu(x + self.conv(x)))

class CNN(nn.Module):
    def __init__(self, d_model, cnn_layers, kernels, strides, channels, padding):
        super().__init__()
        layers = []
        for layer,i,o,k,s,p in zip(cnn_layers, [3] + channels[:-1], channels, kernels, strides, padding):
            layers.append( layer(ni=i, nf=o, ks=k, stride=s, padding=p) )
        self.cnn = nn.Sequential(*layers)
        b,c,h,w = self.cnn(torch.zeros(1,3,48,128)).shape
        print('CNN output = h:{} c:{}'.format(h,c))
#         self.out = nn.Linear(channels[-1]*h, d_model)

    def forward(self, x):
        x = self.cnn(x).permute(0,3,1,2)
        b,w,c,h = x.shape
        return x.view(b,w,-1) # [b,c,h,w]
#         return self.out(x.view(b,w,-1)) # [b,c,h,w]

# Cell
class TransformerModel(nn.Module):

    def __init__(self, nclass=10, nc=3, n_layers=6, d_model=512, d_ff=1024, use_rnn=False, rnn_hidden=256, bidirectional=False):
        super().__init__()
        self.nclass = nclass
        strides = [2, 1, (2,1), 1, (2,1), (2,1), 1]
        channels = [64, 128, 256, 256, 512, 512, 512]
        cnn_layers = [conv_layer] * len(strides)
        kernels = [3] * len(strides)
        padding = [None] * (len(strides)-1) + [0] # None - out size doesnt change
        self.kernels, self.strides, self.channels, self.padding = kernels, strides, channels, padding

        self.cnn = CNN(d_model, cnn_layers, kernels, strides, channels, padding)

        self.transformer = Transformer(nclass, n_layers=n_layers, n_heads=8, d_model=d_model, d_inner=d_ff)

    def forward(self, x, y_input):
        ''' [b,c,h,w], [b,s_d] '''
        x = self.cnn(x) # [b,s_e,512]
        return self.transformer(x, y_input) # [b,s_d,c]

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
def CER(y_pred, y_true):
    # y_pred: [b,s_d,c], y_true: [b,s_d]
    y_pred = y_pred.argmax(-1)
    m = 0
    for yp, yt in zip(y_pred, y_true):
        if yp.shape == torch.Size([]): continue
        yt_text = decode_idxes(yt)
        yp_text = decode_idxes(yp)
        m += wer(yt_text, yp_text)
    return tensor(m / len(y_pred))

# Cell
def WER(y_pred, y_true):
    # y_pred: [b,s_d,c], y_true: [b,s_d]
    y_pred = y_pred.argmax(-1)
    m = 0
    for yp, yt in zip(y_pred, y_true):
        if yp.shape == torch.Size([]): continue
        yt_text = decode_idxes(yt)
        yp_text = decode_idxes(yp)
        m += wer(yt_text.split(' '), yp_text.split(' '))
    return tensor(m / len(y_pred))