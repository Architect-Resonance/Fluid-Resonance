import numpy as np
from scipy import linalg

def build_8x8_ref():
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    adj = np.zeros((8, 8))
    deg = np.zeros(8)
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            adj[i, j] += 1
            adj[j, i] += 1
            deg[i] += 1
            deg[j] += 1
    grounding = np.array([2, 2, 0, 1, 0, 1, 0, 0])
    L = np.diag(deg + grounding) - adj
    return linalg.eigvalsh(L)[0]

def build_7x7_mag(alpha):
    clauses = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 0), (4, 0, 1),
        (5, 0, 3), (6, 2, 4), (7, 5, 6)
    ]
    adj = np.zeros((7, 7), dtype=complex)
    deg = np.zeros(7)
    for c in clauses:
        u, v, w = c
        for i, j in [(u,v), (v,w), (w,u)]:
            if i < 7 and j < 7:
                adj[i, j] += np.exp(1j * alpha)
                adj[j, i] += np.exp(-1j * alpha)
                deg[i] += 1
                deg[j] += 1
    L = np.diag(deg) - adj
    return linalg.eigvalsh(L)[0]

def find_match():
    target = build_8x8_ref()
    print(f"Target λ_min: {target:.10f}")
    
    from scipy.optimize import brentq
    match_a = brentq(lambda a: build_7x7_mag(a) - target, 0, np.pi/2)
    print(f"Match found: alpha = {match_a:.10f} ({match_a/np.pi:.6f} pi)")
    
    # Check if match_a is a "cool" number
    print(f"alpha / (1-ln2) = {match_a / (1-np.log(2)):.6f}")
    print(f"alpha * 7 / pi = {match_a * 7 / np.pi:.6f}")
    print(f"alpha * 4 / pi = {match_a * 4 / np.pi:.6f}") # One octant?

if __name__ == "__main__":
    find_match()
