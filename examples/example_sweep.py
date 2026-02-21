import numpy as np
from src.solver import DirectNavierStokesSolver, SolutionStore

def run_example_sweep():
    """Run a simple parameter sweep for Weissenberg number"""
    
    # Parameters
    base_params = {
        'Nx': 10, 'Ny': 10,
        'F_range': np.linspace(-0.1, 1, 5),
        'Re': 300,
        'delta': 2,
        'a': 0.3, 'b': 0.3, 'd': 1,
        'phi': np.pi/4
    }
    
    # Sweep Weissenberg number
    We_values = [0.2, 0.5, 0.8]
    
    print("="*70)
    print("WILLIAMSON FLUID BIFURCATION ANALYSIS")
    print("="*70)
    print(f"Sweeping Weissenberg number: {We_values}")
    
    for We in We_values:
        print(f"\nSolving for We = {We}")
        print("-"*50)
        
        solver = DirectNavierStokesSolver(
            Nx=base_params['Nx'], Ny=base_params['Ny'],
            Re=base_params['Re'], We=We, delta=base_params['delta'],
            a=base_params['a'], b=base_params['b'], d=base_params['d'],
            phi=base_params['phi'], store_solutions=True
        )
        
        # Solve for different flow rates
        for F in base_params['F_range']:
            solver.F = F
            success = solver.solve()
            print(f"  F = {F:.3f}: {'✓' if success else '✗'}")
    
    print("\n" + "="*70)
    print("SWEEP COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_example_sweep()
