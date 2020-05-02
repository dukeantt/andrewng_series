import numpy as np


class WarmUpExercise:
    def __init__(self, m_size):
        self.identity_size = m_size

    def create_identity_matrix(self):
        # Return 5x5 matrix
        # zero_matrix = np.zeros((matrix_shape, matrix_shape))
        identity_matrix = np.eye(self.identity_size)
        return identity_matrix



