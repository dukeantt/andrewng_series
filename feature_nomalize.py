import numpy as np

class FeatureNomalize:
    def __init__(self):
        pass

    def do_feature_normalization(self,X):
        x = 0
        not_x0 = X[1:3]
        mu = np.mean(not_x0,axis=1)
        sigma = (np.amax(X,axis=1) - np.amin(X,axis=1)).reshape(3,1)[1:3]

