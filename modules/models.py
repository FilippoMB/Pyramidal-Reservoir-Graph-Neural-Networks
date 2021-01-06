import numpy as np
from modules.RGNN import RGNN
from sklearn.linear_model import RidgeClassifier
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import SpectralClustering
import tqdm


class deep(object):
    """
    Compute graph embeddings using parallel stacked of RGNNs
    
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self, T, K, in_scaling, hid_scaling, return_last=True, aggregation='sum', **kwargs):
        self.K             = K
        self.T             = T
        self.return_last   = return_last
        self.aggregation   = aggregation
        self.RGNN_in_list  = [RGNN(input_scaling=in_scaling, **kwargs) for _ in range(self.K)] # The K RGNNs of the first layer (T=0)
        self.RGNN_hid_list = [RGNN(input_scaling=hid_scaling, **kwargs) for _ in range(self.K)] # The K RGNNs used in the next layers (T>0)
        
        # Compute size of output features
        if return_last:
            self.output_size = kwargs['n_internal_units']*K
        else:
            self.output_size = kwargs['n_internal_units']*K*T
            
        
    def get_embeddings(self, L, X):
        """
        L = Laplacian
        X = vertex features
        """
        embeddings_list = []
        for k in range(self.K):
            
            # Embedding from the first layers in the stacks
            out_ = self.RGNN_in_list[k].get_states(L, X)
                        
            if not self.return_last:
                embeddings_list.append(out_)
            
            # Recursively update the embedding in the next layers
            for t in range(1, self.T):
                out_ = self.RGNN_hid_list[k].get_states(L, out_)
                                
                if not self.return_last:
                    embeddings_list.append(out_)
                    
            if self.return_last:
                embeddings_list.append(out_)
                
        # Global pooling
        final_embedding = []
        for emb in embeddings_list:   
            if self.aggregation == 'sum':
                emb = np.sum(emb, axis=0)
            elif self.aggregation == 'average':
                emb = np.mean(emb, axis=0)
            else:
                raise NotImplementedError('unsupported aggregation')   
            final_embedding.append(emb)

        # Concatenate all the layer-wise embeddings
        final_embedding = np.hstack(final_embedding)
        
        return final_embedding


class ARMA(object):
    """    
    Compute graph embeddings using a randomized ARMA architecture
    
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self, T, K, in_scaling, hid_scaling, return_last=None, aggregation='sum', **kwargs):
        self.K             = K
        self.T             = T
        self.aggregation   = aggregation
        self.RGNN_in_list  = [RGNN(input_scaling=in_scaling, **kwargs) for _ in range(self.K)] # The K RGNNs of the first layer (T=0)
        self.RGNN_hid_list = [RGNN(input_scaling=hid_scaling, **kwargs) for _ in range(self.K)] # The K RGNNs used in the next layers (T>0)
        self.return_last   = return_last
            
        # Compute size of output features
        if return_last:
            self.output_size = kwargs['n_internal_units']
        else:
            self.output_size = kwargs['n_internal_units']*T
            
            
    def get_embeddings(self, L, X):
        """
        L = Laplacian
        X = vertex features
        """
        global_embeddings_list = []
        embeddings_list = []
        
        # Embedding from the first ARMA layers
        for k in range(self.K):
            out_k = self.RGNN_in_list[k].get_states(L, X)
            embeddings_list.append(out_k)
        out_ = sum(embeddings_list)
        
        if not self.return_last:
            global_embeddings_list.append(out_)
            
        # Recursively update the embedding
        for t in range(1, self.T):
            embeddings_list = []
            for k in range(self.K):
                out_k = self.RGNN_hid_list[k].get_states(L, out_)
                embeddings_list.append(out_k)
            out_ = sum(embeddings_list)
            
            if not self.return_last:
                global_embeddings_list.append(out_)
                
        if self.return_last:
            global_embeddings_list.append(out_)
              
        # Global pooling
        final_embedding = []
        for emb in global_embeddings_list:   
            if self.aggregation == 'sum':
                emb = np.sum(emb, axis=0)
            elif self.aggregation == 'average':
                emb = np.mean(emb, axis=0)
            else:
                raise NotImplementedError('unsupported aggregation')   
            final_embedding.append(emb)

        # Concatenate all the layer-wise embeddings
        final_embedding = np.hstack(final_embedding)
        
        return final_embedding
          
    
class pool(object):
    """
    Deep architecture interleaved with pooling layers   
    
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self, T, K, in_scaling, hid_scaling, return_last=True, aggregation='sum', **kwargs):
        self.T             = T
        self.K             = K
        self.aggregation   = aggregation
        self.return_last   = return_last        
        self.RGNN_in_list  = [RGNN(input_scaling=in_scaling, **kwargs) for _ in range(self.K)] # The K RGNNs of the first layer (T=0)
        self.RGNN_hid_list = [RGNN(input_scaling=hid_scaling, **kwargs) for _ in range(self.K)] # The K RGNNs used in the next layers (T>0)
        
        # Compute size of output features
        if return_last:
            self.output_size = kwargs['n_internal_units']*K
        else:
            self.output_size = kwargs['n_internal_units']*K*T
            
        
    def get_embeddings(self, L_list, X, D_list):
        """
        L_list = list (pyramid) of coarsened Laplacians. Must be equal to the number T of stacks
        X = vertex features
        D_list = list of pooling matrices. Must be equal to the number T of stacks
        """
        assert len(L_list) == len(D_list) == self.T
        
        embeddings_list = [] 
        for k in range(self.K):
        
            # level 0
            out_ = self.RGNN_in_list[k].get_states(L_list[0], X)
            out_ = D_list[0].dot(out_)
            if not self.return_last:
                embeddings_list.append(out_)
            
            # leveles [1, T-1]
            for t in range(1, self.T):
                out_ = self.RGNN_hid_list[k].get_states(L_list[t], out_)
                out_ = D_list[t].dot(out_)
                if not self.return_last:
                    embeddings_list.append(out_)
                    
            if self.return_last:
                embeddings_list.append(out_)
                
        # Global pooling
        final_embedding = []
        for emb in embeddings_list:   
            if self.aggregation == 'sum':
                emb = np.sum(emb, axis=0)
            elif self.aggregation == 'average':
                emb = np.mean(emb, axis=0)
            else:
                raise NotImplementedError('unsupported aggregation')   
            final_embedding.append(emb)

        # Concatenate all the layer-wise embeddings
        final_embedding = np.hstack(final_embedding)
                            
        return final_embedding
    
    
#####################################################
## WRAPPERS
#####################################################        
class RGNN_classifier():
    """
    Classification model with fit and transform functions
    
    embedding_model = RGNN-based model to compute embeddings {'deep', 'arma', 'pool'}
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    alpha_ridge = regularization factor of the ridge regression classifier
    readout = use a non-linear randomized dense leayer as readout
    readout_units_factor = readout size: (input_size, readout_units_factor*input_size)
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self,
                 embedding_model=None,
                 T=None,
                 K=None,
                 in_scaling=None,
                 hid_scaling=None,
                 return_last=None,
                 aggregation=None,
                 alpha_ridge=None,
                 readout=False,
                 readout_units_factor=None,
                 **kwargs):
    
        if embedding_model == 'deep':
            model = deep
        elif embedding_model == 'arma':
            model = ARMA
        elif embedding_model == 'pool':
            model = pool
        else:
            raise NotImplementedError('unsupported model type')  
        
        # Reservoir-based model
        self.embedding_model = model(K=K, 
                                     T=T, 
                                     in_scaling=in_scaling, 
                                     hid_scaling=hid_scaling, 
                                     return_last=return_last, 
                                     aggregation=aggregation,
                                     **kwargs)
        
        # Readout
        self.readout = readout
        if readout:
            
            # The readout input dimension is the embedding size of the model
            in_dim = self.embedding_model.output_size
                
            # Initialize random readout weights
            W_out = np.random.uniform(low=-1.0, 
                                      high=1.0, 
                                      size=(in_dim, readout_units_factor*in_dim))
            
            # Rescale to unitary norm
            W_out /= np.linalg.norm(W_out) 
            self.W_out = W_out
        else:
            self.W_out = None
            
        # Init the ridge regression classifier
        self.clf = RidgeClassifier(alpha=alpha_ridge)
            
            
    def fit(self, labels, *args):
        print('Fitting model')
                
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        
        # Apply readout if not None
        if self.readout:
            embeddings = np.tanh(embeddings.dot(self.W_out)) 
        
        # Train classifier
        self.clf.fit(embeddings, labels)
        
        # Get training accuracy
        score_tr = self.clf.score(embeddings, labels)
        
        return score_tr
    
    
    def transform(self, labels, *args):
        print('Evaluating model')
        
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        
        # Apply readout if not None
        if self.readout:
            embeddings = np.tanh(embeddings.dot(self.W_out)) 
            
        # Get accuracy
        score_test = self.clf.score(embeddings, labels)
        
        return score_test
    
       
class RGNN_PCA():
    """
    (Kernel) PCA module with fit and transform functions
    
    embedding_model = RGNN-based model to compute embeddings {'deep', 'arma', 'pool'}
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self,
                 embedding_model=None,
                 T=None,
                 K=None,
                 in_scaling=None,
                 hid_scaling=None,
                 return_last=None,
                 aggregation=None,
                 **kwargs):
    
        if embedding_model == 'deep':
            model = deep
        elif embedding_model == 'arma':
            model = ARMA
        elif embedding_model == 'pool':
            model = pool
        else:
            raise NotImplementedError('unsupported model type')  
        
        # Reservoir-based model
        self.embedding_model = model(K=K, 
                                     T=T, 
                                     in_scaling=in_scaling, 
                                     hid_scaling=hid_scaling, 
                                     return_last=return_last, 
                                     aggregation=aggregation,
                                     **kwargs)
        
        self.pca = None
        self.embeddings_tr = None
        
            
    def fit(self, *args):
        print('Fitting model')
                
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        
        self.embeddings_tr = embeddings
                
        # Compute empirical covariance matrix (linear kernel) - Train vs Train
        K_tr = np.dot(self.embeddings_tr, self.embeddings_tr.T)
        self.pca = KernelPCA(n_components=2, kernel='precomputed')
        embeddings_pca = self.pca.fit_transform(K_tr)
        
        # self.pca = umap.UMAP()
        # embeddings = StandardScaler().fit_transform(embeddings)
        # embeddings_pca = self.pca.fit_transform(embeddings)
        
        return embeddings_pca
        
    
    def transform(self, *args):
        print('Evaluating model')
                
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        
        # Compute empirical covariance matrix (linear kernel) - Test vs Train
        K_te = np.dot(embeddings, self.embeddings_tr.T)
        
        embeddings_pca = self.pca.transform(K_te)
        
        return embeddings_pca
    
    
class RGNN_CLUST():
    """
    Spectral clustering module
    
    n_clust = number of clusters to generate
    embedding_model = RGNN-based model to compute embeddings {'deep', 'arma', 'pool'}
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self,
                 n_clust,
                 embedding_model=None,
                 T=None,
                 K=None,
                 in_scaling=None,
                 hid_scaling=None,
                 return_last=None,
                 aggregation=None,
                 **kwargs):
    
        self.n_clust = n_clust
        
        if embedding_model == 'deep':
            model = deep
        elif embedding_model == 'arma':
            model = ARMA
        elif embedding_model == 'pool':
            model = pool
        else:
            raise NotImplementedError('unsupported model type')  
        
        # Reservoir-based model
        self.embedding_model = model(K=K, 
                                     T=T, 
                                     in_scaling=in_scaling, 
                                     hid_scaling=hid_scaling, 
                                     return_last=return_last, 
                                     aggregation=aggregation,
                                     **kwargs)
            
    def fit_transform(self, *args):
        print('Fitting model')
                
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
                        
        clustering = SpectralClustering(n_clusters=self.n_clust, affinity='nearest_neighbors', n_neighbors=7).fit(embeddings)

        return clustering.labels_
        

class RGNN_LDA():
    """
    Spectral clustering module
    
    n_clust = number of clusters to generate
    embedding_model = RGNN-based model to compute embeddings {'deep', 'arma', 'pool'}
    T = depth of each RGNN stack
    K = number of parallel stacks
    in_scaling = scaling of the input weights in the RGNNs in the first layer
    hid_scaling = scaling of the input weights in the RGNNs in all the other layers
    return_last = use as representation only the states from the last layer in the stack
    aggregation = global pooling method to obtain a graph embedding {'sum', 'average'}
    kwargs = dict specifying RGNN hyperparams
    """
    
    def __init__(self,
                 embedding_model=None,
                 T=None,
                 K=None,
                 in_scaling=None,
                 hid_scaling=None,
                 return_last=None,
                 aggregation=None,
                 **kwargs):
    
        
        if embedding_model == 'deep':
            model = deep
        elif embedding_model == 'arma':
            model = ARMA
        elif embedding_model == 'pool':
            model = pool
        else:
            raise NotImplementedError('unsupported model type')  
        
        # Reservoir-based model
        self.embedding_model = model(K=K, 
                                     T=T, 
                                     in_scaling=in_scaling, 
                                     hid_scaling=hid_scaling, 
                                     return_last=return_last, 
                                     aggregation=aggregation,
                                     **kwargs)
        
        self.clf = LinearDiscriminantAnalysis(n_components=2)
            
    def fit_transform(self, labels, *args):
        print('Fitting model')
                
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        
        self.clf.fit(embeddings, labels)
        
        emb_new = self.clf.transform(embeddings)
        return emb_new
        
    def transform(self, *args):
        
        # Generate embeddings
        embeddings = []
        for elem in tqdm.tqdm(zip(*args)):
            emb = self.embedding_model.get_embeddings(*elem)
            embeddings.append(emb)
        embeddings = np.vstack(embeddings)
        
        emb_new = self.clf.transform(embeddings)
        
        return emb_new