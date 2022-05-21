import numpy as np

def square_matrix(a:np.ndarray):
    return True if a.shape[0] == a.shape[1] else False

def upper_triangular(a:np.ndarray):
    return np.allclose(a,np.triu(a),1e-4)

def eigen(a: np.ndarray):
    if not square_matrix(a):
        raise ValueError("matrix must be square.")
    
    eigenvalues = None
    eigenvectors = np.identity(a.shape[0])
    while not upper_triangular(a):
        q,r=np.linalg.qr(a)
        a = np.matmul(np.matmul(q.T, a), q)    
        eigenvectors = np.matmul(eigenvectors,q)

    eigenvalues = a.diagonal()    
    return eigenvalues, eigenvectors


mtrx = np.array([
    [1,4,6],
    [8,9,5],
    [7,2,1],
])


# print(eigen(mtrx)[0])
# print()
# print(eigen(mtrx)[1])
# print()
# print(np.linalg.norm(eigen(mtrx)[1],2,axis=0))
# print()
# print(np.linalg.eig(mtrx)[0])
# print()
# print(np.linalg.eig(mtrx)[1])
# print()
# print(np.linalg.norm(np.linalg.eig(mtrx)[1],2,axis=0))
# print()
# a = np.array([[0.39661415,-0.70654698,0.30693602],
# [0.85680849,0.14902349,-0.76515087],
#  [0.32950937,0.69179719,0.56598094]])
# print(np.linalg.norm(a,2,axis=1))