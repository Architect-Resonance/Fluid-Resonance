import numpy as np
from scipy import linalg

def verify_target():
    L = np.array([
        [ 7, -1, -1, -1, -1, -1,  0,  0],
        [-1,  6, -1, -1, -1,  0,  0,  0],
        [-1, -1,  6, -1, -1,  0, -1,  0],
        [-1, -1, -1,  5, -1, -1,  0,  0],
        [-1, -1, -1, -1,  6,  0, -1,  0],
        [-1,  0,  0, -1,  0,  4, -1, -1],
        [ 0,  0, -1,  0, -1, -1,  4, -1],
        [ 0,  0,  0,  0,  0, -1, -1,  2]
    ])
    
    eigs = linalg.eigvalsh(L)
    print(f"Eigenvalues: {eigs}")
    print(f"λ_min: {eigs[0]:.12f}")
    
    target = 0.4949988739119799
    print(f"Target: {target:.12f}")
    print(f"Error: {eigs[0] - target:.4e}")

if __name__ == "__main__":
    verify_target()
