import numpy as np

def induction_integral(D, r_min=0.01, r_max=1.0, steps=1000):
    """
    Evaluates \int_{r_min}^{r_max} (1/r^2) * r^{D-1} dr
    This represents the induced velocity at a point from a 
    fractal distribution of vorticity with dimension D.
    (Measure d\mu ~ r^{D-1} dr)
    """
    r = np.linspace(r_min, r_max, steps)
    dr = r[1] - r[0]
    # Integrand: (Kernel 1/r^2) * (Volume element r^{D-1})
    # Since u ~ integral (omega x r)/r^3, 
    # it's |u| ~ \int (omega/r^2) * d\mu
    integrand = (1.0 / r**2) * (r**(D-1))
    return np.sum(integrand) * dr

def run_induction_audit():
    print(f"--- S43: INDUCTION KERNEL CONVERGENCE AUDIT ---")
    print(f"{'Dimension D':<15} | {'Integral Value':<20} | {'Status'}")
    print("-" * 55)
    
    for D in [3.0, 2.5, 2.1, 2.0, 1.9, 1.5, 1.0]:
        val = induction_integral(D)
        status = "DIVERGENT" if D <= 2.0 and D > 0 else "CONVERGENT"
        # Wait, the integral is \int r^{D-3} dr.
        # Converges if D-3 > -1 => D > 2.
        # If D < 2, the integral is \int_{r_min} r^{D-3} dr which is r^{D-2}/(D-2).
        # At r=0, this diverges if D-2 < 0 => D < 2.
        # WAIT. 
        # If the dimension is HIGH (D=3), the integral is \int 1 dr = Const. (Convergent at r=0)
        # If the dimension is LOW (D=1), the integral is \int 1/r^2 dr = 1/r. (Divergent at r=0)
        
        # This is COUNTER-INTUITIVE. 
        # Sparser things (D=1) make the kernel MORE singular?
        # Yes, because if you pack all vorticity onto a line, the LOCAL density is infinite.
        # But for blow-up, we need the TOTAL induced strain to blow up.
        
        print(f"{D:<15.1f} | {val:<20.4f} | {status}")

    print("-" * 55)
    print("INSIGHT: The '1D Singularity' (vortex line) is the most dangerous.")
    print("But CKN proved that the 1D measure of singularities is ZERO.")
    print("This implies the fluid MUST stay 2D or 3D in its concentration.")

if __name__ == "__main__":
    run_induction_audit()
