import json
import numpy as np

def summarize():
    with open('helicity_ratio_results.json', 'r') as f:
        data = json.load(f)
        
    print(f"{'Rho':<6} | {'Init s':<8} | {'Final s':<8} | {'Avg E_sol':<12} | {'Avg Z':<10}")
    print("-" * 55)
    for k in sorted(data.keys(), key=float):
        s_vals = [x['s'] for x in data[k]]
        e_vals = [x['e_sol'] for x in data[k]]
        z_vals = [x['z'] for x in data[k]]
        print(f"{k:<6} | {s_vals[0]:<8.4f} | {s_vals[-1]:<8.4f} | {np.mean(e_vals):<12.2e} | {np.mean(z_vals):<10.2f}")

if __name__ == "__main__":
    summarize()
