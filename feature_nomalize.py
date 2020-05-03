import numpy as np

class FeatureNomalize:
    def __init__(self):
        pass

    def do_feature_normalization(self,X):
        not_x0 = X[1:3]
        mu = np.mean(not_x0,axis=1).reshape(2,1)
        sigma = (np.amax(X,axis=1) - np.amin(X,axis=1)).reshape(3,1)[1:3]
        not_x0 = (not_x0 - mu)/sigma
        return np.vstack((X[:1], not_x0))

