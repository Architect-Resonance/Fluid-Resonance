import numpy as np

def count_fano_modes():
    k_range = range(-4, 5) # Enough to cover kmag <= 3.5
    modes = []
    for kx in k_range:
        for ky in k_range:
            for kz in k_range:
                kmag2 = kx**2 + ky**2 + kz**2
                if 0 < kmag2 <= 12.25: # |k| <= 3.5
                    modes.append((kx, ky, kz))
    
    # Group by GF(2)^3
    classes = {}
    for k in modes:
        c = (k[0] % 2, k[1] % 2, k[2] % 2)
        classes[c] = classes.get(c, 0) + 1
    
    print(f"Total modes: {len(modes)}")
    print("GF(2)^3 Classes (mod 2):")
    for c, count in sorted(classes.items()):
        name = "Diagonal (1,1,1)" if c == (1,1,1) else ("Even (0,0,0)" if c == (0,0,0) else "Fano point")
        print(f"  {c}: {count} ({name})")
    
    fano_counts = [count for c, count in classes.items() if c != (0,0,0)]
    print(f"\nFano class counts: {fano_counts}")
    print(f"Diagonal (1,1,1) count: {classes.get((1,1,1), 0)}")
    
    others = [count for c, count in classes.items() if c != (1,1,1) and c != (0,0,0)]
    print(f"Other 6 Fano classes: {others}")

if __name__ == "__main__":
    count_fano_modes()
