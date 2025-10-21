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
    def runtest(self):
        k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.mesh.material_map_cylindrical_centered,
            298.15
        )
        print(k_matrix)
    def runtest2(self):
        #print(self.mesh.r_nodes)
        #print(self.mesh.r_faces)
        #print(self.mesh.r_centers)
        #np.savetxt("r_mapping.csv", self.mesh.material_map_cylindrical_centered)
        #np.savetxt("r_nodes.csv", self.mesh.r_nodes)
        #np.savetxt("r_centers.csv", self.mesh.r_centers)
        #np.savetxt("r_faces.csv", self.mesh.r_faces)
        #arr=self.mesh.material_map_cylindrical_centered
        #print(arr[2])
        print(self.mesh.r_nodes.shape)
        print(self.mesh.r_faces.shape)
        print(self.mesh.r_centers.shape)
        print(self.mesh.starting_index)
        if True:
            print(self.mesh.glass_kanthal_cement_interface)
            print(self.mesh.kanthal_cement_interface)
            print(self.mesh.cement_ceramic_interface)
            print(self.mesh.ceramic_reflective_interface)
            print(self.mesh.reflective_outer_interface)
            #print(self.mesh.cold_heat_interface_front)
            #print(self.mesh.cold_heat_interface_back)
        if False:
            print(self.mesh.kanthal_r_position_start)
            print(self.mesh.kanthal_r_position_end)
            print(self.mesh.kanthal_z_position_start)
            print(self.mesh.kanthal_z_position_end)

        if False:
            print(self.mesh.sample_radius)
            print(self.mesh.glass_outer_radius)
            print(self.mesh.kanthal_outer_radius)
            print(self.mesh.cement_outer_radius)
            print(self.mesh.ceramic_outer_radius)
            print(self.mesh.reflective_outer_radius)
            print(self.mesh.inner_glass_tube_area)
            print(self.mesh.sample_air_space_volume)
            print(self.mesh.cubic_inner_surface)
            print(self.mesh.cubic_outer_surface)
    def runtest3(self):
        #np.savetxt("r_scenters.csv", self.solver.r_centers)
        #np.savetxt("r_sfaces.csv", self.solver.r_faces)
        #print(self.solver.r_centers.shape)
        #print(self.solver.r_faces.shape)
        #print(self.solver.dr_centers.shape)
        if False:
            pd.DataFrame(self.solver.dr_grid).to_csv("dr_grid.csv", index=False)
            pd.DataFrame(self.solver.dr_centers).to_csv("dr_centers.csv", index=False)
            pd.DataFrame(self.solver.dz_centers).to_csv("dz_centers.csv", index=False)
            pd.DataFrame(self.solver.dr_grid).to_csv("dr_grid.csv", index=False)
            pd.DataFrame(self.solver.dz_grid).to_csv("dz_grid.csv", index=False)
            pd.DataFrame(self.solver.dr1_grid).to_csv("dr1_grid.csv", index=False)
            pd.DataFrame(self.solver.dr2_grid).to_csv("dr2_grid.csv", index=False)
            pd.DataFrame(self.solver.dz1_grid).to_csv("dz1_grid.csv", index=False)
            pd.DataFrame(self.solver.dz2_grid).to_csv("dz2_grid.csv", index=False)
        if True:
            """
            k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.mesh.material_map_cylindrical_centered,
            298.15
            )
            """
            #pd.DataFrame(k_matrix).to_csv("k_matrix.csv", index=False)
            pd.DataFrame(self.solver.dr1_grid).to_csv("dr1_grid.csv", index=False)
            #pd.DataFrame(self.solver.dr2_grid).to_csv("dr2_grid.csv", index=False)
            pd.DataFrame(self.solver.r_faces).to_csv("r_faces.csv", index=False)
            pd.DataFrame(self.solver.r_centers).to_csv("r_centers.csv", index=False)
            #arr = self.solver.harmonic_mean_nonuniform(k_matrix[:-1, :], k_matrix[1:, :], self.solver.dr1_grid, self.solver.dr2_grid)
            #pd.DataFrame(arr).to_csv("k_harmonic_mean.csv", index=False)

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
    #test_instance.find_num_nodes_for_stretch(0.0157, 0.1175, 0.0001275*2, 1.05, 200)