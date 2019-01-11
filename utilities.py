import progressbar
import pandas as pd
import gensim
import os
import numpy as np
from itertools import tee, islice
import pickle

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def one_hot(arr, size):
    onehot = np.zeros((len(arr),size), dtype = int)
    for i in range(len(arr)):
        if not np.isnan(arr[i]):            
            onehot[i, int(arr[i])]=1
    #onehot[np.arange(len(arr)), arr] =1
    return onehot

def bow_to_ohv(dct):
    '''converts bag of words to one-hot format.
    dct: {hadm_id: (X, y)}
    '''
    xy = {}
    for h in dct.keys():
        X, y = [], dct[h][1]
        for t in dct[h][0]:
            X.append(np.array( [(lambda x: 1 if xx > 0 else 0)(xx) for xx in t] ) )
        X = np.array(X)
        xy[h] = (X,y)
    return xy

def bow_sampler(x, size):
    if not pd.isnull(x).all():
        bow = np.sum(one_hot(x, size), axis=0) 
        bow = np.array([(lambda x: 1 if x >0 else 0)(xx) for xx in bow])
        first = one_hot(x,size)[0]
        last = one_hot(x,size)[-1]
        return [first, bow, last]
    else:
        return np.nan

def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
    
### Progressbar tools ### 
def make_widget():
    widgets = [progressbar.Percentage(), ' ', progressbar.SimpleProgress(), ' ', 
                                 progressbar.Bar(left = '[', right = ']'), ' ', progressbar.ETA(), ' ', 
                                 progressbar.DynamicMessage('LOSS'), ' ',  progressbar.DynamicMessage('PREC'), ' ',
                                 progressbar.DynamicMessage('REC')]
    bar = progressbar.ProgressBar(widgets = widgets)
    return bar


### Find nearest timestamps ###
def find_prev(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == 0:
        return array[idx]
    else:
        return array[idx-1]

def find_next(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if idx == len(array) -1:
        return array[idx]
    else:
        return array[idx+1]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#### Skip Gram ##### 
def skip_gram(corpus):
    corpus = list(corpus)
    SG = gensim.models.Word2Vec(sentences = corpus, sg = 1, size = 200, window = 5, min_count = 0, hs = 1, negative = 0)
    weights = SG.wv.syn0
    vocab = dict([(k, v.index) for k, v in SG.wv.vocab.items()])
    w2i, i2w = vocab_index(vocab)
    
    #turn sentences into word vectors for each admission
    corpus = [list(map(lambda i: w2i[i] if i in w2i.keys() else 0, vv)) for vv in corpus]    
    #word vectors here
    w2v = [] 
    
    for sentence in corpus:
        one_hot = np.zeros((len(sentence), weights.shape[0]))
        one_hot[np.arange(len(sentence)), sentence] = 1
        one_hot = np.sum(one_hot, axis= 0)
        w2v.append(np.dot(one_hot.reshape(one_hot.shape[0]), weights))
    return w2v, SG, weights, vocab

def vocab_index (vocab):
    word2idx = vocab
    idx2word = dict([(v,k) for k, v in vocab.items()])
    return (word2idx, idx2word)


### utility function ###

def flatten(lst):
    make_flat = lambda l: [item for sublist in l for item in sublist]
    return make_flat(lst)    

def balanced_subsample(x,y,subsample_size=1.0):
    class_xs = []
    min_elems = None
    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]
    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)
    xs = []
    ys = []
    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys

def hierarchical_subsample(x, dx, y,subsample_size=1.0):
    from sklearn.utils import shuffle
    class_xs = []
    class_dxs = []
    min_elems_x = None
    min_elems_d = None

    for yi in np.unique(y):
        elems_x = x[(y == yi)]
        elems_d = dx[(y==yi)]
        class_xs.append((yi, elems_x))
        class_dxs.append((yi, elems_d))
        if min_elems_x == None or elems_x.shape[0] < min_elems_x:
            min_elems_x = elems_x.shape[0]
            min_elems_d = elems_d.shape[0]

    use_elems_x = min_elems_x
    use_elems_d = min_elems_d
    if subsample_size < 1:
        use_elems_x = int(min_elems_x*subsample_size)
        use_elems_d = int(min_elems_d*subsample_size)

    xs = []
    dxs = []
    ys = []

    for lst1, lst2 in zip(class_xs, class_dxs):
        ci = lst1[0]
        this_xs = lst1[1]
        this_dxs = lst2[1]
        
        if len(this_xs) > use_elems_x:
            this_xs, this_dxs = shuffle(this_xs, this_dxs)

        x_ = this_xs[:use_elems_x]
        d_ = this_dxs[:use_elems_d]
        y_ = np.empty(use_elems_x)
        y_.fill(ci)

        xs.append(x_)
        dxs.append(d_)
        ys.append(y_)

    xs = np.concatenate(xs)
    dxs = np.concatenate(dxs)
    ys = np.concatenate(ys)

    return xs, dxs, ys

#### Pickling Tools ####

def large_save(dct, file_path):
    '''dct: {k: v}
    '''
    lst = sorted(dct.keys())
    chunksize =10000
    #chunk bytes
    bytes_out= bytearray(0)
    for idx in range(0, len(lst), chunksize):
        bytes_out += pickle.dumps(dict([(k,v) for k,v in dct.items() if k in lst[idx: idx+ chunksize]]))
    with open(file_path, 'wb') as f_out:
            for idx in range(0, len(bytes_out), chunksize):
                f_out.write(bytes_out[idx:idx+chunksize])
    #split files
    for idx in range(0, len(lst), chunksize):
        chunk = dict([(k,v) for k,v in dct.items() if k in lst[idx: idx+ chunksize]])
        with open(file_path+'features_'+str(idx+chunksize), 'wb') as f_out:
            pickle.dump(chunk, f_out, protocol=2)

def large_read(file_path):
    import os.path
    bytes_in = bytearray(0)
    max_bytes = int(1e5)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data

def create_dir_if_not_exists(directory):
    """Creates a directory if it doesn't already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)