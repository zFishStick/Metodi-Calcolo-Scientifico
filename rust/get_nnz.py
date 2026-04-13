import sys
from scipy.io import mmread
from scipy.sparse.linalg import splu

path = sys.argv[1]
A = mmread(path).tocsc() # type: ignore

LU = splu(A)
print(LU.nnz)