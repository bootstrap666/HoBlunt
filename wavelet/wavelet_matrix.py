import numpy as np
import pywt


class Wavelet_matrix:
    def __init__(self,vector_length,wavelet='haar'):

        if not isinstance(vector_length, (int, np.integer)) or isinstance(vector_length, bool):
            raise TypeError("vector_length_level must be an integer")

        if vector_length < 1:
            raise ValueError("vector_length must be greater than zero")

        if vector_length & (vector_length - 1):
            raise ValueError("vector_length must be a power of 2")

        self.max_level = pywt.dwt_max_level(vector_length, wavelet)
        self.wavelet = wavelet
        self.vector_length = vector_length

        self.transform_matrix = np.zeros((self.vector_length,self.vector_length), dtype=float)

        for k in range(self.vector_length):
            self.transform_matrix[k,:] = self._wavelet_vector(k)

    def _wavelet_vector(self, k):
        e_k = np.zeros(self.vector_length)
        e_k[k] = 1.0
        
        coeffs = pywt.wavedec(e_k, self.wavelet, mode='periodization', level=self.max_level)
        return np.concatenate(coeffs)
    
    def get_basis_vector(self, k):
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise TypeError("k must be an integer")
        
        if not 0 <= k < self.vector_length:
            raise IndexError(
                f"Basis vector index {k} out of range [0, {self.transform_matrix.shape[0] - 1}]"
            )

        return self.transform_matrix[:, k]

    def children_of(self,k):
        pass
        
    def parents_of(self,k):
        pass
        
    def find_first_index_in_level(self,level):
        pass

    def find_indexes_level(self, level):
        pass

    def get_level(self, k):
        pass