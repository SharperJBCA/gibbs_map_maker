import numpy as np 

class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        self.vector = np.zeros(cols) # this is the x vector
        self.data = data if data is not None else {} # this is an ancillary data needed for the matrix

    def forward(self, vector):
        """This is the A.dot(x) method"""
        if vector.shape[0] != self.cols:
            raise ValueError("Dimension mismatch: {} != {}".format(vector.shape[0], self.cols))
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def backward(self, vector):
        """This is the A.T.dot(x) method"""
        if vector.shape[0] != self.rows:
            raise ValueError("Dimension mismatch: {} != {}".format(vector.shape[0], self.rows))
        raise NotImplementedError("Subclasses must implement the backward method")
    
    def bkwdfwd(self, vector):
        """This is the A.T.dot(A.dot(x)) method"""
        return self.backward(self.forward(vector))
    

class PointingMatrix(Matrix):
    """Bin TOD into pixels
    
    data = {'pixels':np.ndarray, 'npix':int}
    """

    def forward(self, vector):
        return vector[self.data['pixels']]
    
    def backward(self, vector):
        pixel_edges = np.arange(self.data['npix'] + 1) 
        top = np.histogram(self.data['pixels'], bins=pixel_edges, weights=vector)[0]
        return top

class OffsetMatrix(Matrix):
    """Bin TOD into offsets 
    
    data = {'offsets':np.ndarray, 'noff':int}
    """
    def forward(self, vector):
        return vector[self.data['offsets']]
    
    def backward(self, vector):
        offset_edges = np.arange(self.data['noff'] + 1) 
        top = np.histogram(self.data['offsets'], bins=offset_edges, weights=vector)[0]
        return top
    
class GradientMatrix(Matrix):
    """Bin TOD into offsets 
    
    data = {'coordinate':np.ndarray}
    """
    def forward(self, vector):
        return self.data['coordinate']*vector[0]
    
    def backward(self, vector):
        return np.array([np.sum(vector*self.data['coordinate'])])


class MatrixA(Matrix):
    """Test class to check gibbs sampler works
    
    matrix = [[3, 1],
                [1, 2]]
    """
    def forward(self, vector):
        return np.array([3 * vector[0] + vector[1], vector[0] + 2 * vector[1]])

    def backward(self, vector):
        return np.array([3 * vector[0] + vector[1], vector[0] + 2 * vector[1]])
    

class MatrixA2(Matrix):
    """Test class to check gibbs sampler works
    matrix = [[1, 2],
                [2, 1],
                [1, 1]]
    """
    def forward(self, vector):
        A = np.array([[1, 2], 
                      [2, 1], 
                      [1, 1],
                        [1, 2],
                        [2, 1]]).astype(np.float64)
        return A.dot(vector)

    def backward(self, vector):
        A = np.array([[1, 2], 
                      [2, 1], 
                      [1, 1],
                        [1, 2],
                        [2, 1]]).astype(np.float64)
        return A.T.dot(vector)

class MatrixB2(Matrix):
    """Test class to check gibbs sampler works
    matrix = [[3, 2, 1],
                [1, 2, 3],
                [2, 1, 3]]
    """
    def forward(self, vector):
        B = np.array([[3, 2, 1], 
                      [1, 2, 3], 
                      [2, 1, 3],
                      [3, 2, 1],
                      [1, 2, 3]]).astype(np.float64)
        return B.dot(vector)

    def backward(self, vector):
        B = np.array([[3, 2, 1], 
                      [1, 2, 3], 
                      [2, 1, 3],
                      [3, 2, 1],
                      [1, 2, 3]]).astype(np.float64)
        return B.T.dot(vector)
