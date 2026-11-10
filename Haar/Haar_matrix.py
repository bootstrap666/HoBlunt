import numpy as np

class Haar_matrix:
    def __init__(self,max_level):

        if not isinstance(max_level, (int, np.integer)) or isinstance(max_level, bool):
            raise TypeError("max_level must be an integer")

        if max_level < 1:
            raise ValueError("max_level must be greater than zero")

        self.max_level = max_level

        N = 2**max_level
        self.transform_matrix = np.zeros((N,N), dtype=float)

        for k in range(N):
            self.transform_matrix[:, k] = self._haar_vector(N, k)

    def _haar_vector(self, N, k):
        if k == 0:
            return np.ones(N, dtype=float) / np.sqrt(N)

        # Finds if k vector is in S_0, S_1, ... S_{maxlevel -1}
        level = self.get_level(k) #int(np.ceil(np.log2(k + 1)))
        # Within S_j, determines the vector index
        first_index = 2**(level - 1)
        position = k - first_index

        # Determines the decimation and the shift of the basis function
        L = N // (2**(level - 1))
        p = position * L

        basis_vector = np.zeros(N)
        basis_vector[p:p + L//2] = 1 / np.sqrt(L)
        basis_vector[p + L//2:p + L] = -1 / np.sqrt(L)

        return basis_vector
    
    def get_basis_vector(self, k):
        if not isinstance(max_level, (int, np.integer)) or isinstance(max_level, bool):
            raise TypeError("max_level must be an integer")
        
        if not 0 <= k < 2**self.max_level:
            raise IndexError(
                f"Basis vector index {k} out of range [0, {self.transform_matrix.shape[0] - 1}]"
            )

        return self.transform_matrix[:, k]

    def children_of(self,k):
    # Here, k is the global index of the control basis function. 
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise TypeError("k must be an integer")
        if not 0 <= k < 2**self.max_level:
            raise IndexError(
                f"Basis vector index {k} out of range [0, {self.transform_matrix.shape[0] - 1}]"
            )

        children = []
        k_level = self.get_level(k) #int(np.ceil(np.log2(k + 1)))

        # Equation (12) requires the local index of k-th control basis function
        k_local_index = k - self.find_first_index_in_level(k_level)

        for level in range(k_level+1, self.max_level+1):
            first = self.find_first_index_in_level(level)
            children.extend(
                # Note that in Equation (12), the upper limit is a <= so I ommited the -1 at the upper limit
                first + j for j in range(2**(level-k_level)*k_local_index,2**(level-k_level)*(k_local_index+1))
            )
        return children

    def parents_of(self,k):
    # Here, k is the global index of the control basis function. 
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise TypeError("k must be an integer")
        if not 0 <= k < 2**self.max_level:
            raise IndexError(
                f"Basis vector index {k} out of range [0, {self.transform_matrix.shape[0] - 1}]"
            )

        parents = []
        k_level = self.get_level(k) #int(np.ceil(np.log2(k + 1)))

        # Equation (14) requires the local index of k-th control basis function
        k_local_index = k - self.find_first_index_in_level(k_level)

        for level in range(k_level):
            first = self.find_first_index_in_level(level)
            parents.append(
                first + int(np.floor(k_local_index/(2**(k_level-level)))) 
            )
        return parents

    def find_first_index_in_level(self,level):
        if not isinstance(level, (int, np.integer)) or isinstance(level, bool):
            raise TypeError("level must be an integer")

        if not 0 <= level <= self.max_level:
            raise ValueError(
                f"level must be between 0 and {self.max_level}, got {level}"
            )
            
        return 0 if level == 0 else 2**(level - 1)

    def find_indexes_level(self, level):
        if not isinstance(level, (int, np.integer)) or isinstance(level, bool):
            raise TypeError("level must be an integer")
            
        if not 0 <= level <= self.max_level:
            raise ValueError(
                f"level must be between 0 and {self.max_level}, got {level}"
            )
    
        return list(range(
            self.find_first_index_in_level(level),
            self.find_first_index_in_level(level + 1)
        ))

    def get_level(self, k):
        return int(np.ceil(np.log2(k + 1)))