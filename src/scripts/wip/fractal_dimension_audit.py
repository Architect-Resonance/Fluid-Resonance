import numpy as np

def fractal_dimension_audit(enstrophy_field, thresholds):
    """
    Estimates the Fractal Dimension (Box-counting proxy)
    of the set where Z > threshold.
    """
    dims = []
    for T in thresholds:
        active_cells = np.sum(enstrophy_field > T)
        # In a 3D box, if Z is concentrated on a line, N(L) ~ 1/L
        # Dim = log(N) / log(1/L)
        # We use a simplified proxy: log(active_cells) / log(total_cells)
        # This is very rough, but shows the 'concentration' index
        dim_proxy = 3.0 * np.log(active_cells + 1e-9) / np.log(len(enstrophy_field) + 1e-9)
        dims.append(dim_proxy)
    return dims

def run_fractal_audit():
    print(f"--- S42: FRACTAL DIMENSION & SCALAR GRADIENT AUDIT ---")
    print(f"{'Concentration Level':<20} | {'Active Cells':<15} | {'Dim Proxy'}")
    print("-" * 55)
    
    # Grid: 100x100x100 = 1e6 cells
    total_cells = 1000000
    grid = np.zeros(total_cells)
    
    # Case A: Uniform Flow (D=3)
    grid_a = np.ones(total_cells)
    
    # Case B: Vortex Sheet (D=2)
    # 100x100 cells active
    grid_b = np.zeros(total_cells)
    grid_b[:10000] = 1.0
    
    # Case C: Vortex Line (D=1)
    # 100 cells active
    grid_c = np.zeros(total_cells)
    grid_c[:100] = 1.0
    
    # Case D: Point Singularity (D=0)
    grid_d = np.zeros(total_cells)
    grid_d[0] = 1.0
    
    for name, g in [("Uniform", grid_a), ("Sheet", grid_b), ("Line", grid_c), ("Point", grid_d)]:
        dims = fractal_dimension_audit(g, [0.5])
        print(f"{name:<20} | {int(np.sum(g > 0.5)):<15} | {dims[0]:.4f}")

    print("-" * 55)
    print("INSIGHT: As D decreases, the 'Connectivity' of the interaction graph collapses.")
    print("If D=1 (Vortex line), the interaction is too sparse for O(N^2) blow-up.")

if __name__ == "__main__":
    run_fractal_audit()
