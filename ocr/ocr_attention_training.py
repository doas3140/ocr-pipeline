# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/07_ocr_attention_training.ipynb (unless otherwise specified).

__all__ = ['PAD', 'PAD', 'attention_config', 'allowed_chars', 'allowed_fonts', 'TextlineProcessor', 'TextlineAndFont',
           'TextlineList', 'im2seq_data_collate', 'train_transforms', 'valid_transforms', 'create_data', 'conv_output',
           'MultiHeadAttention', 'feed_forward', 'EncoderBlock', 'DecoderBlock', 'get_output_mask',
           'PositionalEncoding', 'TransformerEmbedding', 'compose', 'TransformerEncoder', 'CNN', 'RevConv',
           'get_normal_cnn', 'get_partially_rev_cnn', 'AttentionModel']

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
from .ocr_dataset_brno import create_df as create_brno_df
from .ocr_dataset_sroie2019 import sroie_ocr_config, char_freq
from .ocr_dataset_fontsynth import fontsynth_config, char_freq
from .ocr_dataset_brno import brno_ocr_config
PAD = sroie_ocr_config.PAD # PAD - how much is data padded
PAD = 0

# Cell
allowed_chars = {'N', '3', 'V', 'P', '7', '1', '#', '9', '"', 'C', 'Q', 'B', 'E', '>', '@', ',', 'M', '{', ']',
                 ';', '^', "'", '&', '6', 'Z', '*', '<', '+', 'G', 'X', '!', ':', '-', '[', '|', '$', '5', 'I',
                 'H', '=', 'Y', '.', 'R', 'S', '/', 'T', '}', 'K', '0', '?', 'U', ')', '_', 'D', 'J', 'L', '4',
                 'W', '%', '(', ' ', 'F', '8', '~', '\\', 'A', '2', 'O'}

# allowed_chars = fontsynth_config.allowed_chars

allowed_fonts = ['Unknown', 'Andale_Mono', 'Arial', 'Arial_Black', 'Arial_Bold', 'Arial_Bold_Italic', 'Arial_Italic',
'Comic_Sans_MS_Bold', 'Courier_New', 'Courier_New_Bold', 'Courier_New_Bold_Italic', 'Courier_New_Italic',
'Georgia', 'Georgia_Bold', 'Georgia_Bold_Italic', 'Georgia_Italic', 'Impact', 'Times_New_Roman',
'Times_New_Roman_Bold', 'Times_New_Roman_Bold_Italic', 'Times_New_Roman_Italic', 'Trebuchet_MS',
'Trebuchet_MS_Bold', 'Trebuchet_MS_Bold_Italic', 'Trebuchet_MS_Italic', 'Verdana', 'Verdana_Bold',
'Verdana_Bold_Italic', 'Verdana_Italic', 'brno_easy', 'brno_medium', 'sroie2019', 'Comic_Sans_MS']

class attention_config:
    LINE_HEIGHT = 48
    USE_DEFAULT_CLASSES = True
    label_delim = '`'
    pad_idx = 0 # aka: label_delim idx
    allowed_chars = allowed_chars
    allowed_fonts = allowed_fonts

# Cell
from .ocr_crnn_training import TextlineProcessor, TextlineAndFont, TextlineList, MyImageList, im2seq_data_collate
from .ocr_crnn_training import one_hot_text, decode_single_ctc, decode_ctc, gaussian_blur, rand_resize

# Cell
class TextlineProcessor(TextlineProcessor):
    def __init__(self, ds:ItemList):
        self.create_classes(ds.classes, ds.font_classes)
        self.use_default_classes = attention_config.USE_DEFAULT_CLASSES
        self.default_classes = attention_config.allowed_chars
        self.default_font_classes = attention_config.allowed_fonts

    def create_classes(self, classes, font_classes):
        self.classes, self.font_classes = classes, font_classes
        if classes is not None:
            self.classes = [attention_config.label_delim] + classes
            self.c2i = {v:k for k,v in enumerate(self.classes)}
            self.f2i = {v:k for k,v in enumerate(font_classes)}

# Cell
class TextlineAndFont(ItemBase):
    ''' F = font, S = string
    data: tensor(S), tensor(F)
    obj: str(S), str(F)
    raw: str(S), list(F)
    '''
    def __init__(self, data, obj, raw):self.data, self.obj, self.raw = data, obj, raw
    def __str__(self, n=20):
        string = self.obj[0][:n]+['...'] if len(self.obj[0]) > n else self.obj[0]
        return self.obj[1][:5] +'...'+ attention_config.label_delim.join([str(o) for o in string])
    def __hash__(self): return hash(str(self))

# Cell
class TextlineList(ItemList):
    _processor = TextlineProcessor
    def __init__(self, items:Iterator, classes=None, font_classes=None, label_delim:str=None, one_hot:bool=False, **kwargs):
        self.classes = classes
        self.font_classes = font_classes
        items = [(string.split(attention_config.label_delim),font) for string,font in items] # CHANGED
        super().__init__(items, **kwargs)
        self.processor = [TextlineProcessor(self)]

    def get(self, i):
        stridxs, fontidx = self.items[i] # int, list of ints
        return TextlineAndFont( (tensor(stridxs), tensor(fontidx)),
                                ([self.classes[c] for c in stridxs], self.font_classes[fontidx]), self.items[i])

    def analyze_pred(self, nn_output, thresh=0.5, _=None):
        font_pred, y_pred = nn_output # [c1], [s_e,c2]
        assert len(listify(y_pred.shape)) == 2 # (no batch inputs)
        return font_pred.argmax(dim=-1), decode_single_ctc(y_pred.argmax(dim=-1)), _, _ # [1], [seq_len], _, _

    def reconstruct(self, data_out):
        fontidx, t_argmax, _, lengths = data_out # output from data / output from nn_out -> analyze_pred
        stridxs = [int(i) for i in t_argmax]
        fontidx = int(fontidx)
        return TextlineAndFont((one_hot_text(stridxs, self.c), fontidx),
                                ([self.classes[c] for c in stridxs], self.font_classes[fontidx]), data_out)

    @property
    def c(self): return len(self.classes)

# Cell
def im2seq_data_collate(batch:ItemsList, pad_idx:int=0)->Tensor:
    if isinstance(batch[0][1], int): return data_collate(batch)
    "Convert `batch` items to tensor data."
    data = to_data(batch) # list of (image, text) pairs
    # image: [3,48,w], text: [n,c], where n's and w's are different
    max_w = max([image.shape[2] for image, (text,font) in data])
    max_h = max([image.shape[1] for image, (text,font) in data])
    max_n = max([text.shape[0] for image, (text,font) in data])
#     _, num_classes = data[0][1].shape

    images = torch.zeros(len(batch), 3, max_h, max_w)
    fonts = torch.zeros(len(batch)).long()
#     texts = torch.zeros(len(batch), max_n, num_classes)
    texts = []
    nn_out_seq_len, texts_len = [], []
    for i, (image, (text,font)) in enumerate(data):
        fonts[i] = font
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
    return images, (fonts, texts, tensor(nn_out_seq_len).type(torch.int), tensor(texts_len).type(torch.int))

# Cell
train_transforms = [
    rand_resize(pad=(0,PAD), p=1.0),
    rotate(degrees=(-2, 2), p=0.6),
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
    data = (MyImageList.from_df(df, path='.', cols='image_path')
        .split_from_df(col='valid')
        .label_from_df(cols='string', label_cls=TextlineList, label_delim=attention_config.label_delim)
        .transform((train_transforms, valid_transforms), tfm_y=False)
        .databunch(bs=bs, collate_fn=partial(im2seq_data_collate, pad_idx=attention_config.pad_idx))
        .normalize()
    )
#     data.train_dl.numworkers=0
#     data.valid_dl.numworkers=0

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
def feed_forward(d_model:int, d_ff:int, ff_p:float=0., activ_func=partial(nn.ReLU, inplace=True), double_drop:bool=True):
    ''' [...,d] -> [...,d] '''
    layers = [nn.Linear(d_model, d_ff), activ_func()]
    if double_drop: layers.append(nn.Dropout(ff_p))
    return SequentialEx(*layers, nn.Linear(d_ff, d_model), nn.Dropout(ff_p), MergeLayer(), nn.LayerNorm(d_model))

# Cell
class EncoderBlock(nn.Module):
    "Encoder block of a Transformer model."
    def __init__(self, n_heads, d_model, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, p=p, bias=bias, scale=scale)
        self.ff  = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)

    def forward(self, x, mask=None):
        ''' [b,s_e,512], [1,1,s_e,s_e] -> [b,s_e,512] '''
        return self.ff(self.mha(x, x, mask=mask))

# Cell
class DecoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    def __init__(self, n_heads, d_model, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, p=p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)

    def forward(self, x, enc, mask_out=None):
        ''' [b,s_d,512], [b,s_e,512], [1,1,s_d,s_d] -> [b,s_d,512] '''
        return self.ff(self.mha2(self.mha1(x, x, mask_out), enc))

# Cell
def get_output_mask(inp, pad_idx=1):
    ''' [b,s_e,...] -> [1,1,s_e,s_e] '''
    return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].type(torch.bool)

# Cell
class PositionalEncoding(Module):
    "Encode the position with a sinusoid."
    def __init__(self, d:int): self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))

    def forward(self, pos:Tensor):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc

# Cell
class TransformerEmbedding(nn.Module):
    "Embedding + positional encoding + dropout"
    def __init__(self, emb_sz, vocab_sz=None, drop=0.):
        super().__init__()
        self.emb_sz = emb_sz
        if vocab_sz is None: self.embed = None
        else: self.embed = nn.Embedding(vocab_sz, emb_sz)
        self.pos_enc = PositionalEncoding(emb_sz)
        self.drop = nn.Dropout(drop)
        self.alpha = nn.Parameter(tensor(1.))

    def forward(self, inp):
        ''' [] -> [] '''
        pos = torch.arange(0, inp.size(1), device=inp.device).float()
        if self.embed is not None: inp = self.embed(inp)
        return self.drop(inp + self.alpha * self.pos_enc(pos))
#         return self.drop(inp * math.sqrt(self.emb_sz) + self.pos_enc(pos))

# Cell
def compose(funcs):
    def func_out(x, *args):
        for f in listify(funcs):
            x = f(x, *args)
        return x
    return func_out

# Cell
class TransformerEncoder(Module):
    def __init__(self, n_layers=6, n_heads=8, d_model=512, d_inner=1024, p=0.1,
                 bias=True, scale=True, double_drop=True):
        self.enc_emb = TransformerEmbedding(d_model, drop=0.) # no need to embed encoding from cnn output
        args = (n_heads, d_model, d_inner, p, bias, scale, double_drop)
        self.encoders = nn.ModuleList([EncoderBlock(*args) for _ in range(n_layers)])

    def forward(self, inp):
        ''' [b,s_e,512], [b,s_d,c] or [b,s_d] -> [b,s_d,c] (c - num classes) '''
        enc = self.enc_emb(inp) #
        return compose(self.encoders)(enc) # [b,s_e,512]

# Cell
class CNN(nn.Module):
    def __init__(self, d_model, cnn_layers, kernels, strides, channels, padding, nc=3):
        super().__init__()
        layers = []
        for layer,i,o,k,s,p in zip(cnn_layers, [nc] + channels[:-1], channels, kernels, strides, padding):
            layers.append( layer(ni=i, nf=o, ks=k, stride=s, padding=p) )
        self.cnn = nn.Sequential(*layers)
        b,c,h,w = self.cnn(torch.zeros(1,3,48,128)).shape
        self.out = nn.Linear(h*c, d_model)
        print('CNN output = h:{} c:{}'.format(h,c))

    def forward(self, x):
        x = self.cnn(x).permute(0,3,1,2)
        b,w,c,h = x.shape
        return self.out(x.view(b,w,-1)) # [b,c,h,w]

# Cell
import revtorch as rv

def RevConv(ni, nf, ks, stride, padding):
    assert ni == nf and stride == 1
    f_func = conv_layer(ni//2, nf//2, ks, stride=stride, padding=padding)
    g_func = conv_layer(ni//2, nf//2, ks, stride=stride, padding=padding)
    layers = nn.ModuleList([rv.ReversibleBlock(f_func, g_func)])
    return rv.ReversibleSequence(layers, eagerly_discard_variables = True)

# Cell
def get_normal_cnn(dx=1):
    strides = [2, 1, (2,1), 1, (2,1), 1, (2,1), 1]
    channels = [int(c*dx) for c in [64, 64, 128, 128, 256, 256, 512, 512]]
    cnn_layers = [conv_layer] * len(strides)
    kernels = [3] * len(strides)
    padding = [None] * len(strides) # None - out size doesnt change
    return cnn_layers, channels, kernels, strides, padding

# Cell
def get_partially_rev_cnn(dx=1):
    strides = [2, 1, (2,1), 1, (2,1), 1, (2,1), 1]
    channels = [int(c*dx) for c in [64, 64, 128, 128, 256, 256, 512, 512]]
    cnn_layers = [conv_layer, RevConv] * (len(strides)//2)
    kernels = [3] * len(strides)
    padding = [None] * len(strides) # None - out size doesnt change
    return cnn_layers, channels, kernels, strides, padding

# Cell
class AttentionModel(nn.Module):

    def __init__(self, nclass=10, fclass=10, nc=3, n_layers=6, d_model=512, d_ff=1024, use_rnn=False, bidirectional=False):
        super().__init__()

        cnn_layers, self.channels, self.kernels, self.strides, self.padding = get_partially_rev_cnn(dx=1/2)
        self.cnn = CNN(d_model, cnn_layers, self.kernels, self.strides, self.channels, self.padding, nc=nc)

        # font prediction
        h,w = 2,d_model
        self.adaptive_pool = nn.AdaptiveAvgPool2d([h,w]) # this([h,w])([B,H,W]) -> [B,h,w]
        f_model = 2 # font embedding
        self.font_ff = nn.Sequential(nn.Linear(h*w, fclass*f_model), nn.ReLU())
        self.font_out = nn.Linear(f_model, 1)
#         self.font_emb = nn.Linear(f_model, d_model)

        # text prediction
        self.transformer = TransformerEncoder(n_layers=n_layers, n_heads=8, d_model=d_model, d_inner=d_ff)

        self.use_rnn = use_rnn
        if self.use_rnn:
            self.rnn = nn.LSTM(d_model, d_model, bidirectional=bidirectional)
            mult = 1 if not bidirectional else 2
            d_model = d_model * mult

        self.out = nn.Linear(d_model, nclass)

        self.nclass, self.d_model, self.fclass, self.f_model = nclass, d_model, fclass, f_model

    def forward(self, x):
        ''' [b,c,h,w], [b,s_d] '''
        b,c,h,w = x.shape
        x = self.cnn(x) # [b,w,512]
        f = self.adaptive_pool(x).view(b,-1) # [b,h_a x w_a] (_a = adaptive pool params)
        f_enc = self.font_ff(f).view(-1, self.fclass, self.f_model)
        f_out = self.font_out(f_enc).view(-1, self.fclass)
#         f_emb = self.font_emb(f_enc) # [b,f,512]
        x = self.transformer(x)
        if self.use_rnn: x, _ = self.rnn(x)
        return f_out, self.out(x)

# Cell
from .ocr_crnn_training import CTCFontLoss, AddLossMetrics, WordErrorRate