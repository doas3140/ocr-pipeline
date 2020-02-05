# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/10_nlp_transformer.ipynb (unless otherwise specified).

__all__ = ['readdf', 'readdfsentences', 'd']

# Cell
def readdf(filename):
    ''' read file to dataframe '''
    f = open(filename)
    data, sentence, label = [], [], []
    sentence_idx = 0
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                for word, tag in zip(sentence, label):
                    data.append( (word, tag, sentence_idx) )
                sentence_idx += 1
                sentence, label = [], []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        for word, tag in zip(sentence, label):
            if tag != '':
                data.append( (word, tag, sentence_idx) )
    return pd.DataFrame(data, columns=['word', 'tag', 'sentence_idx'])

# Cell
def readdfsentences(document_data, valid_pct=0.2):
    data = []
    for i,document in enumerate(document_data):
        words = [word for word,tag in zip(*document)]
        tags = [tag for word,tag in zip(*document)]
        data.append( (' '.join(words), ' '.join(tags), len(document_data)*valid_pct > i) )
    return pd.DataFrame(data, columns=['sentences', 'labels', 'valid'])

d = readdfsentences(document_data)
d.head()