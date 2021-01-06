import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from spektral.utils.logging import log, init_logging
from spektral.utils.convolution import normalized_adjacency
from spektral.datasets.tud import load_data
from modules.pooling import  preprocess
from modules.models import RGNN_classifier


PRECOMPUTED = False                             # Load precomputed pooled graphs
N_CONFIGS=20                                    # Number of random configurations
N_EVALS=3                                       # Number of evaluations of each configuration
N_SPLITS=5                                      # Number of outer folds
IN_KF_SPLITS=1                                  # Number of inner folds. If 1, random 90/10 split
spec_rad_range = [.1, .9]                       # Range of uniform distribution from where the spectral radius is drawn
in_scaling_range = [.1, .8]                     # Range of uniform distribution from where the input scaling is drawn
hid_scaling_range = [.1, .8]                    # Range of uniform distribution from where the input scaling of the hidden gESN is drawn
ridge_alphas = [1e2, 1e1, 1e0, 1e-1, 1e-2]      # Draw the ridge regression regularization at random from this list

P = {
    'MODE':'pool',                              # Strategy to generate embeddings: {'deep', 'arma', 'pool'}
    'T': 3,                                     # Number of stacked layers
    'K': 1,                                     # Number of parallel gESN in each module
    'n_internal_units': 50,                     # Number of reservoir units in each gESN
    'connectivity': 1,                          # Connectivity of reservoir. If <1, the rervoir weights matrix is sparse
    'input_weights_mode': 'uniform',            # Options: {'very_sparse', 'sparse_uniform', 'binomial', 'uniform'}
    'readout': False,                           # Use an ELM readout if True
    'readout_units_factor': None,               # Multiplicative factor of the number of units in the ELM readout
    'max_norm': None,                           # Reservoir max L2 norm (if None, provide spectral_radius)
    'pool': 'decim',                            # Pooling strategy (only if MODE == 'pool). Options: {'graclus', 'nmf', 'decim'}
    'coarse_level': [0, 1, 2],                  # Defines the pooling levels (only if MODE == 'pool). It should be consisntent with T
    'return_last': True,                        # Compute embeddings only from the output of the last layer
    'aggregation': 'sum',                       # Global pooling strategy {'sum', 'average'}
    'dataset_ID': 'MUTAG',                      # PROTEINS','ENZYMES','NCI1','MUTAG','Mutagenicity','DD','COLLAB','REDDIT-BINARY'
    }

# Load dataset
if P['dataset_ID'] == 'synth':
    loaded = np.load('data/hard.npz', allow_pickle=True)
    X_train, A_train, y_train = loaded['tr_feat'], list(loaded['tr_adj']), loaded['tr_class']
    X_test, A_test, y_test    = loaded['te_feat'], list(loaded['te_adj']), loaded['te_class']
    X = np.hstack((X_train, X_test))
    A = A_train + A_test
    y = np.vstack((y_train, y_test))
    
else:
    A, X, y = load_data(P['dataset_ID'], normalize_features=True, clean=False)
    X = np.array(X)
y = np.argmax(y, axis=-1)
    
if P['MODE'] == 'pool':
    
    if PRECOMPUTED:
        L, D = pickle.load(open('data/'+P['dataset_ID']+"_"+P['pool']+".pkl","rb"))
        
    else:
        A_pooled, X, D = preprocess(A, X, coarsening_levels=P['coarse_level'], pool=P['pool'])
        L = [[normalized_adjacency(a_).astype(np.float32) for a_ in A_] for A_ in A_pooled]
    
else:
    L = [normalized_adjacency(a_) for a_ in A]
    

df_out = None # init pandas df to write restuls
log_dir = init_logging()  # Create log directory and file
results = {'train_acc': [], 'test_acc': [], 'tr_time': [], 'test_time': []}

# Outer fold
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True) # init k-fold
for train_index, test_index in kf.split(X, y):
    
    # Outer split (test)
    X_train_out, y_train_out = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    L_train_out = [L[idx] for idx in train_index]
    L_test = [L[idx] for idx in test_index]
    if P['MODE'] == 'pool':
        D_train_out = [D[idx] for idx in train_index]
        D_test = [D[idx] for idx in test_index]
    
    # Init containers for the best results
    best_hyperparams = {'spec_rad': None, 
                        'in_scaling': None,
                        'hid_scaling': None,
                        'alpha': None}
    best_val_score = 0
    best_config = None
    best_seed = None
    
    # Try all the random configs
    for i, c in enumerate(range(N_CONFIGS)):
        
        # set the seed
        np.random.seed(i)
        
        # Generate a new random config
        spec_rad = np.random.uniform(low=spec_rad_range[0], high=spec_rad_range[1])
        in_scaling = np.random.uniform(low=in_scaling_range[0], high=in_scaling_range[1])
        hid_scaling = np.random.uniform(low=hid_scaling_range[0], high=hid_scaling_range[1])
        alpha = np.random.choice(ridge_alphas)
        
        # Store the config in a dictionary
        C = {'spec_rad': spec_rad,
             'in_scaling': in_scaling,
             'hid_scaling': hid_scaling,
             'alpha': alpha }
        
        # Initialize the gESN configuration dictionary
        gESN_dict = {
                'n_internal_units': P['n_internal_units'],
                 'spectral_radius': C['spec_rad'], 
                 'connectivity': P['connectivity'], 
                 'leak': None,
                 'noise_level': 0.0, 
                 'circle': False,
                 'input_weights_mode': P['input_weights_mode']}
        
        # Evaluate different random initializations of the reservoir
        e_scores = []
        for e in range(N_EVALS):
            
            # Init model
            m = RGNN_classifier(embedding_model=P['MODE'],
                         K=P['K'],
                         T=P['T'],
                         in_scaling=C['in_scaling'],
                         hid_scaling=C['hid_scaling'],
                         alpha_ridge=C['alpha'],
                         readout=P['readout'],
                         readout_units_factor=P['readout_units_factor'],
                         return_last=P['return_last'],
                         aggregation=P['aggregation'],
                         **gESN_dict)
        
            # Inner fold
            if IN_KF_SPLITS != 1:
                kf = StratifiedKFold(n_splits=IN_KF_SPLITS, shuffle=True) # init k-fold
                for in_tr_index, val_index in kf.split(X_train_out, y_train_out):
                    
                    # Inner split (val)
                    X_train_in, y_train_in = X_train_out[in_tr_index], y_train_out[in_tr_index]
                    X_val, y_val = X_train_out[val_index], y_train_out[val_index]
                    L_train_in = [L_train_out[idx] for idx in in_tr_index]
                    L_val = [L_train_out[idx] for idx in val_index]
                    if P['MODE'] == 'pool':
                        D_train_in = [D_train_out[idx] for idx in in_tr_index]
                        D_val = [D_train_out[idx] for idx in val_index]
                    
                    # Fit model and test model
                    if P['MODE'] == 'pool':
                        tr_score = m.fit(y_train_in, L_train_in, X_train_in, D_train_in)
                        val_score = m.transform(y_val, L_val, X_val, D_val)
                    else:
                        tr_score = m.fit(y_train_in, L_train_in, X_train_in)
                        val_score = m.transform(y_val, L_val, X_val)
                    e_scores.append(val_score)
                    
            else:
                indices = list(range(X_train_out.shape[0]))
                in_tr_index, val_index = train_test_split(indices, stratify=y_train_out, test_size=0.1)
                
                # Inner split (val)
                X_train_in, y_train_in = X_train_out[in_tr_index], y_train_out[in_tr_index]
                X_val, y_val = X_train_out[val_index], y_train_out[val_index]
                L_train_in = [L_train_out[idx] for idx in in_tr_index]
                L_val = [L_train_out[idx] for idx in val_index]
                if P['MODE'] == 'pool':
                    D_train_in = [D_train_out[idx] for idx in in_tr_index]
                    D_val = [D_train_out[idx] for idx in val_index]
                
                # Fit model and test model
                if P['MODE'] == 'pool':
                    tr_score = m.fit(y_train_in, L_train_in, X_train_in, D_train_in)
                    val_score = m.transform(y_val, L_val, X_val, D_val)
                else:
                    tr_score = m.fit(y_train_in, L_train_in, X_train_in)
                    val_score = m.transform(y_val, L_val, X_val)
                e_scores.append(val_score)
                
               
        # Compute mean on all evals and save if is better
        mean_val_score = np.mean(e_scores)   
        if mean_val_score > best_val_score:
            best_val_score = mean_val_score
            best_config = C
            best_seed = i
            
    
    # Define gESN parameters for test
    gESN_dict = {
                'n_internal_units': P['n_internal_units'],
                 'spectral_radius': best_config['spec_rad'], 
                 'connectivity': P['connectivity'], 
                 'leak': None,
                 'noise_level': 0.0, 
                 'circle': False,
                 'input_weights_mode': P['input_weights_mode']}
    
    # Restore the seed
    np.random.seed(best_seed)
    
    # Evaluate different random initializations of the reservoir
    e_scores_train = []
    e_scores_test = []
    e_times_train = []
    e_times_test = []
    for e in range(N_EVALS):
        
        # Init model for test on the outer fold
        m = RGNN_classifier(embedding_model=P['MODE'],
                             K=P['K'],
                             T=P['T'],
                             in_scaling=best_config['in_scaling'],
                             hid_scaling=best_config['hid_scaling'],
                             alpha_ridge=best_config['alpha'],
                             readout=P['readout'],
                             readout_units_factor=P['readout_units_factor'],
                             return_last=P['return_last'],
                             aggregation=P['aggregation'],
                             **gESN_dict)
        
        # Fit and test the model
        if P['MODE'] == 'pool':
            start_time = time.time()
            tr_score = m.fit(y_train_out, L_train_out, X_train_out, D_train_out)
            tr_time = time.time() - start_time
            
            start_time = time.time()
            test_score = m.transform(y_test, L_test, X_test, D_test)
            test_time = time.time() - start_time
        else:
            start_time = time.time()
            tr_score = m.fit(y_train_out, L_train_out, X_train_out)
            tr_time = time.time() - start_time
            
            start_time = time.time()
            test_score = m.transform(y_test, L_test, X_test)
            test_time = time.time() - start_time
        
        # Store results
        e_scores_train.append(tr_score)
        e_scores_test.append(test_score)
        e_times_train.append(tr_time)
        e_times_test.append(test_time)
        
    # Store aggregated results on the outer fold
    results['train_acc'].append(np.mean(e_scores_train))
    results['test_acc'].append(np.mean(e_scores_test))
    results['tr_time'].append(np.mean(e_times_train))
    results['test_time'].append(np.mean(e_times_test))

P['train_acc_mean'] = np.mean(results['train_acc'])
P['train_acc_std']  = np.std(results['train_acc'])
P['test_acc_mean']  = np.mean(results['test_acc'])
P['test_acc_std']   = np.std(results['test_acc'])
P['tr_time_mean']   = np.mean(results['tr_time'])
P['tr_time_std']    = np.std(results['tr_time'])
P['test_time_mean'] = np.mean(results['test_time'])
P['test_time_std']  = np.std(results['test_time'])
log(P)

if df_out is None:
    df_out = pd.DataFrame([P])
else:
    df_out = pd.concat([df_out, pd.DataFrame([P])])
df_out.to_csv(log_dir + 'results.csv')