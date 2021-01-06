import numpy as np
import time
from modules.pooling import preprocess
from sklearn.model_selection import train_test_split
from spektral.utils.convolution import normalized_adjacency
from spektral.datasets.tud import load_data
import matplotlib.pyplot as plt
from modules.models import RGNN_LDA
np.random.seed(0)

######## HYPERPARAMS ########
P = {
    'MODE':'pool',                      # Strategy to generate embeddings: {'deep', 'arma', 'pool'}
    'T': 2,                             # Number of stacked layers
    'K': 1,                             # Number of parallel gESN in each module
    'n_internal_units': 100,            # Number of reservoir units in each gESN
    'spectral_radius': .9,              # Reservoir spectral radius (if None, provide max_norm)
    'max_norm': None,                   # Reservoir max L2 norm (if None, provide spectral_radius)
    'connectivity': 1,                  # Connectivity of reservoir. If <1, the rervoir weights matrix is sparse
    'in_scaling': .5,                   # Scaling of the input weights in the first layer
    'hid_scaling': .8,                  # Scaling of the input weights in the hidden layers
    'input_weights_mode': 'uniform',    # Options: {'very_sparse', 'sparse_uniform', 'binomial', 'uniform'}
    'return_last': False,               # Use only the output of the last layer to compute the embeddings
    'aggregation': 'sum',               # Global pooling aggregation
    'pool': 'decim',                    # Pooling strategy (ignored if MODE != 'pool). Options: {'graclus', 'nmf', 'decim'}
    'coarse_level': [0, 1],             # Defines the pooling levels (only if MODE == 'pool). It should be consisntent with T
    'dataset_ID': "synth",              # TUD datasets IDs or "synth"
}
print(P)


######## LOAD DATA ########
if P['dataset_ID'] == 'synth':
    loaded = np.load('data/easy.npz', allow_pickle=True)
    X_train, A_train, y_train = loaded['tr_feat'], list(loaded['tr_adj']), loaded['tr_class']
    X_test, A_test, y_test    = loaded['te_feat'], list(loaded['te_adj']), loaded['te_class']
    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)
else:    
    A, X, y = load_data(P['dataset_ID'], normalize_features=True, clean=False)
    y = np.argmax(y, axis=-1)
    A_train, A_test, \
    X_train, X_test, \
    y_train, y_test = train_test_split(A, X, y, test_size=0.1, stratify=y)

if P['MODE'] == 'pool':
    print('Computing coarsened Laplacians')
    A_train, X_train, D_train = preprocess(A_train, X_train, coarsening_levels=P['coarse_level'], pool=P['pool'])
    A_test, X_test, D_test = preprocess(A_test, X_test, coarsening_levels=P['coarse_level'], pool=P['pool'])
    
    L_train = [[normalized_adjacency(a_).astype(np.float32) for a_ in A] for A in A_train]
    L_test = [[normalized_adjacency(a_).astype(np.float32) for a_ in A] for A in A_test]
       
else:
    L_train = [normalized_adjacency(A_) for A_ in A_train]
    L_test  = [normalized_adjacency(A_) for A_ in A_test]
    
L = L_train+L_test
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
if P['MODE'] == 'pool':
    D = D_train + D_test # np.concatenate((D_train, D_test))


# Dictionary that specifies the gESN config
gESN_dict = {
        'n_internal_units': P['n_internal_units'],
        'spectral_radius': P['spectral_radius'], 
        'max_norm': P['max_norm'], 
        'leak': None,
        'connectivity': P['connectivity'], 
        'noise_level': 0.0, 
        'circle': False,
        'input_weights_mode': P['input_weights_mode']}

################# LDA
m = RGNN_LDA(embedding_model=P['MODE'],
              K=P['K'],
              T=P['T'],
              in_scaling=P['in_scaling'],
              hid_scaling=P['hid_scaling'],
              return_last=P['return_last'],
              aggregation=P['aggregation'],
              **gESN_dict)

start_time = time.time()
if P['MODE'] == 'pool':
    X_lda = m.fit_transform(y, L, X, D)
 
else:
    X_lda = m.fit_transform(y, L, X)
end_time = time.time() - start_time
print('time: {:.1f}'.format(end_time))

f, ax = plt.subplots(figsize=(4,4))
ax.scatter(X_lda[:,0], X_lda[:,1], c=y, s=1, cmap='Paired')
plt.show()