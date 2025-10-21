import numpy as np
import config
from materials import TubeFurnaceMaterials
from mesh import TubeFurnaceMesh
from solver import TubeFurnaceHeatSolver
import pandas as pd
from scipy.optimize import brentq

class Test:
    def __init__(self):
        self.materials = TubeFurnaceMaterials()
        self.mesh = TubeFurnaceMesh()
        self.solver = TubeFurnaceHeatSolver()
        self.sample_radius = config.GLASS_TUBE_INNER_RADIUS
        self.glass_outer_radius = config.GLASS_TUBE_OUTER_RADIUS
        self.kanthal_outer_radius = config.HEATING_COIL_RADIUS
        self.cement_outer_radius = config.GLASS_TUBE_OUTER_RADIUS + config.FURNACE_CEMENT_THICKNESS
        self.ceramic_outer_radius = config.INSULATION_OUTER_RADIUS
        self.reflective_outer_radius = config.REFLECTIVE_CASING_OUTER_RADIUS


        pd.DataFrame(self.solver.dr1_grid).to_csv("dr1_grid.csv", index=False)
        pd.DataFrame(self.solver.dr2_grid).to_csv("dr2_grid.csv", index=False)
        pd.DataFrame(self.solver.r_faces).to_csv("r_.csv", index=False)
        pd.DataFrame(self.solver.r_centers).to_csv("r_centers.csv", index=False)

    def create_hyperbolic_grid(self, start, end, num_points, stretch_factor):
        """Generates a grid stretched towards the start point using hyperbolic tangent."""
        linear_space = np.linspace(0, 1, num_points)
        L = end - start
        stretched_space = 0.5 * (1 - np.tanh(stretch_factor * (1 - 2 * linear_space)) / np.tanh(stretch_factor))
        grid = start + L * stretched_space
        return grid

    def find_num_nodes_for_stretch(self, start, end, target_dr, stretch_factor_limit, initial_nodes=20, max_iter=100):
        num_points = initial_nodes
        for i in range(max_iter):
            if num_points <= 1:
                num_points += 1
                continue

            grid = test_instance.create_hyperbolic_grid(start, end, num_points, stretch_factor_limit)
            dr = np.diff(grid)

            if dr[0] <= target_dr:
                print(f"Required number of nodes found: {num_points}")
                print(f"Initial spacing achieved: {dr[0]:.4e}")
                print(f"Max-to-min ratio: {dr.max()/dr.min():.2f}")
                return num_points, grid

            num_points += 1

        print("Could not find a suitable number of nodes within max iterations.")
        return None, None

    def find_optimal_stretch_factor(self, start, end, num_points, target_dr):
        def objective_function(beta):
            grid = test_instance.create_hyperbolic_grid(start, end, num_points, beta)
            first_dr = grid[1] - grid[0]
            return first_dr - target_dr
        
        # Use brentq to find the optimal stretch factor
        optimal_beta = brentq(objective_function, 1.0, 20.0)
        return optimal_beta

if __name__ == "__main__":
    # Create an instance of the Test class
    test_instance = Test()
    
    # Call the method from the instance
    test_instance.runtest2()
    test_instance.find_num_nodes_for_stretch(0.0157, 0.1175, 0.0001275*2, 1.05, 200)