import numpy as np

def count_fano_modes():
    k_range = range(-3, 4)
    modes = []
    for kx in k_range:
        for ky in k_range:
            for kz in k_range:
                kmag2 = kx**2 + ky**2 + kz**2
                if 0 < kmag2 <= 9: # |k| <= 3
                    modes.append((kx, ky, kz))
    
    # Group by GF(2)^3
    classes = {}
    for k in modes:
        c = (k[0] % 2, k[1] % 2, k[2] % 2)
        classes[c] = classes.get(c, 0) + 1
    
    print(f"Total modes: {len(modes)}")
    print("GF(2)^3 Classes (mod 2):")
    for c, count in sorted(classes.items()):
        print(f"  {c}: {count}")
    
    # Identify the (1,1,1) class
    diag_count = classes.get((1,1,1), 0)
    others = [count for c, count in classes.items() if c != (1,1,1) and c != (0,0,0)]
    
    print(f"\nDiagonal (1,1,1) count: {diag_count}")
    print(f"Other Fano classes: {others}")
    print(f"Mean of others: {np.mean(others)}")
    print(f"Even class (0,0,0) count: {classes.get((0,0,0), 0)}")

if __name__ == "__main__":
    count_fano_modes()
