import os
import numpy as np
import config
import pandas as pd
from scipy.optimize import brentq

class HyperbolicGridGenerator:
    def __init__(self):
        # Extracting dimensional parameters from config
        self.sample_radius = config.GLASS_TUBE_INNER_RADIUS
        self.glass_outer_radius = config.GLASS_TUBE_OUTER_RADIUS
        self.kanthal_outer_radius = config.HEATING_COIL_RADIUS
        self.cement_outer_radius = config.GLASS_TUBE_OUTER_RADIUS + config.FURNACE_CEMENT_THICKNESS
        self.ceramic_outer_radius = config.INSULATION_OUTER_RADIUS
        self.reflective_outer_radius = config.REFLECTIVE_CASING_OUTER_RADIUS

        self.target_radius = config.HEATING_COIL_WIRE_DIAMETER/config.RADIAL_NODES_KANTHAL
        # Optimizing each layer
        self.glass_grid, self.glass_num_nodes, self.glass_minmax_ratio, self.glass_min_dr, self.glass_max_dr, self.glass_min_diff_dr, self.glass_max_diff_dr = self.find_num_nodes_for_stretch(self.sample_radius, self.glass_outer_radius, self.target_radius, config.GLASS_STRETCH_FACTOR)
        self.cement_grid, self.cement_num_nodes, self.cement_minmax_ratio, self.cement_min_dr, self.cement_max_dr, self.cement_min_diff_dr, self.cement_max_diff_dr = self.find_num_nodes_for_stretch(self.kanthal_outer_radius, self.cement_outer_radius, self.target_radius, config.CEMENT_STRETCH_FACTOR)
        self.ceramic_grid, self.ceramic_num_nodes, self.ceramic_minmax_ratio, self.ceramic_min_dr, self.ceramic_max_dr, self.ceramic_min_diff_dr, self.ceramic_max_diff_dr = self.find_num_nodes_for_stretch(self.cement_outer_radius, self.ceramic_outer_radius, self.target_radius, config.CERAMIC_STRETCH_FACTOR)
        self.reflective_grid, self.reflective_num_nodes, self.reflective_minmax_ratio, self.reflective_min_dr, self.reflective_max_dr, self.reflective_min_diff_dr, self.reflective_max_diff_dr = self.find_num_nodes_for_stretch(self.ceramic_outer_radius, self.reflective_outer_radius, self.target_radius, config.REFLECTIVE_STRETCH_FACTOR)
        self.total_num_nodes = sum([self.glass_num_nodes, self.cement_num_nodes, self.ceramic_num_nodes, self.reflective_num_nodes])
        # Statistics
        self.glass_nodes_without_stretching = round((self.glass_outer_radius - self.sample_radius)/self.target_radius)
        self.cement_nodes_without_stretching = round((self.cement_outer_radius - self.kanthal_outer_radius)/self.target_radius)
        self.ceramic_nodes_without_stretching = round((self.ceramic_outer_radius - self.cement_outer_radius)/self.target_radius)
        self.reflective_nodes_without_stretching = round((self.reflective_outer_radius - self.ceramic_outer_radius)/self.target_radius)
        self.total_nodes_without_stretching = sum([self.glass_nodes_without_stretching, self.cement_nodes_without_stretching, self.ceramic_nodes_without_stretching, self.reflective_nodes_without_stretching])
        # Compression ratios
        self.glass_compression_ratio = self.glass_nodes_without_stretching / self.glass_num_nodes
        self.cement_compression_ratio = self.cement_nodes_without_stretching / self.cement_num_nodes
        self.ceramic_compression_ratio = self.ceramic_nodes_without_stretching / self.ceramic_num_nodes
        self.reflective_compression_ratio = self.reflective_nodes_without_stretching / self.reflective_num_nodes
        self.total_compression_ratio = self.total_nodes_without_stretching / self.total_num_nodes
        # Excluded reflective casing
        self.total_num_nodes_without_reflective = sum([self.glass_num_nodes, self.cement_num_nodes, self.ceramic_num_nodes])
        self.total_nodes_without_stretching_without_reflective = sum([self.glass_nodes_without_stretching, self.cement_nodes_without_stretching, self.ceramic_nodes_without_stretching])
        self.total_compression_ratio_without_reflective = self.total_nodes_without_stretching_without_reflective / (self.total_num_nodes - self.reflective_num_nodes)

    def create_hyperbolic_grid(self, start, end, num_points, stretch_factor):
        """Generates a grid stretched towards the start point using hyperbolic tangent."""
        linear_space = np.linspace(0, 1, num_points)
        L = end - start
        stretched_space = 0.5 * (1 - np.tanh(stretch_factor * (1 - 2 * linear_space)) / np.tanh(stretch_factor))
        grid = start + L * stretched_space
        return grid

    def find_num_nodes_for_stretch(self, start, end, target_dr, stretch_factor_limit, initial_nodes=3, max_iter=1000):
        num_points = initial_nodes
        for i in range(max_iter):
            if num_points <= 1:
                num_points += 1
                continue

            grid = self.create_hyperbolic_grid(start, end, num_points, stretch_factor_limit)
            dr = np.diff(grid)

            if dr[0] <= target_dr:
                minmax_ratio = dr.max()/dr.min()    
                #print(f"Required number of nodes found: {num_points}")
                #print(f"Initial spacing achieved: {dr[0]:.4e}")
                #print(f"Max-to-min ratio: {dr.max()/dr.min():.2f}")
                return dr, num_points, dr.max()/dr.min(), dr.min(), dr.max(), abs(np.diff(dr)).min(), np.diff(dr).max()

            num_points += 1

        print("Could not find a suitable number of nodes within max iterations.")
        return None, None

    def find_optimal_stretch_factor(self, start, end, num_points, target_dr):
        def objective_function(beta):
            grid = self.create_hyperbolic_grid(start, end, num_points, beta)
            first_dr = grid[1] - grid[0]
            return first_dr - target_dr
        
        # Use brentq to find the optimal stretch factor
        optimal_beta = brentq(objective_function, 1.0, 20.0)
        return optimal_beta

    def print_summary(self):
        print("Finding optimal stretch factors and number of nodes for each layer with hyperbolic tangent stretching:")
        print(f"    - Target radius: {self.target_radius*1000:.4f} mm")
        if True:
            print("----------------------------------------------------------")
            print(f"    - Layer 1: Sample to glass: {self.sample_radius*1000:.4f}mm to {self.glass_outer_radius*1000:.4f}mm ({(self.glass_outer_radius - self.sample_radius)*1000:.4f} mm)")
            print(f"    - Layer 2: Glass to kanthal: {self.glass_outer_radius*1000:.4f}mm to {self.kanthal_outer_radius*1000:.4f}mm ({(self.kanthal_outer_radius - self.glass_outer_radius)*1000:.4f} mm)")
            print(f"    - Layer 3: Kanthal to cement: {self.kanthal_outer_radius*1000:.4f}mm to {self.cement_outer_radius*1000:.4f}mm ({(self.cement_outer_radius - self.kanthal_outer_radius)*1000:.4f} mm)")
            print(f"    - Layer 4: Cement to ceramic: {self.cement_outer_radius*1000:.4f}mm to {self.ceramic_outer_radius*1000:.4f}mm ({(self.ceramic_outer_radius - self.cement_outer_radius)*1000:.4f} mm)")
            print(f"    - Layer 5: Ceramic to reflective: {self.ceramic_outer_radius*1000:.4f}mm to {self.reflective_outer_radius*1000:.4f}mm ({(self.reflective_outer_radius - self.ceramic_outer_radius)*1000:.4f} mm)")
        print("----------------------------------------------------------")
        print(f"    - Layer 2: Glass: {self.sample_radius*1000:.4f}mm to {self.glass_outer_radius*1000:.4f}mm ({(self.glass_outer_radius - self.sample_radius)*1000:.4f} mm)")
        print(f"    - Optimized stretch factor: {config.GLASS_STRETCH_FACTOR}")
        print(f"    - Number of nodes: {self.glass_num_nodes} (Compared to: {self.glass_nodes_without_stretching}), Compression ratio: {self.glass_compression_ratio:.4f}")
        print(f"    - Min-to-max ratio: {self.glass_minmax_ratio:.4f}")
        print(f"    - Min spacing: {self.glass_min_dr*1000:.4f} mm, Max spacing: {self.glass_max_dr*1000:.4f} mm")
        print(f"    - Min diff spacing: {self.glass_min_diff_dr*1000:.4f} mm, Max diff spacing: {self.glass_max_diff_dr*1000:.4f} mm")
        print("----------------------------------------------------------")
        print(f"    - Layer 3: Cement: {self.kanthal_outer_radius*1000:.4f}mm to {self.cement_outer_radius*1000:.4f}mm ({(self.cement_outer_radius - self.kanthal_outer_radius)*1000:.4f} mm)")
        print(f"    - Optimized stretch factor: {config.CEMENT_STRETCH_FACTOR}")
        print(f"    - Number of nodes: {self.cement_num_nodes} (Compared to: {self.cement_nodes_without_stretching}), Compression ratio: {self.cement_compression_ratio:.4f}")
        print(f"    - Min-to-max ratio: {self.cement_minmax_ratio:.4f}")
        print(f"    - Min spacing: {self.cement_min_dr*1000:.4f} mm, Max spacing: {self.cement_max_dr*1000:.4f} mm")
        print(f"    - Min diff spacing: {self.cement_min_diff_dr*1000:.4f} mm, Max diff spacing: {self.cement_max_diff_dr*1000:.4f} mm")
        print("----------------------------------------------------------")
        print(f"    - Layer 4: Ceramic: {self.cement_outer_radius*1000:.4f}mm to {self.ceramic_outer_radius*1000:.4f}mm ({(self.ceramic_outer_radius - self.cement_outer_radius)*1000:.4f} mm)")
        print(f"    - Optimized stretch factor: {config.CERAMIC_STRETCH_FACTOR}")
        print(f"    - Number of nodes: {self.ceramic_num_nodes} (Compared to: {self.ceramic_nodes_without_stretching}), Compression ratio: {self.ceramic_compression_ratio:.4f}")
        print(f"    - Min-to-max ratio: {self.ceramic_minmax_ratio:.4f}")
        print(f"    - Min spacing: {self.ceramic_min_dr*1000:.4f} mm, Max spacing: {self.ceramic_max_dr*1000:.4f} mm")
        print(f"    - Min diff spacing: {self.ceramic_min_diff_dr*1000:.4f} mm, Max diff spacing: {self.ceramic_max_diff_dr*1000:.4f} mm")
        if False:
            print("----------------------------------------------------------")
            print(f"    - Layer 5: Reflective Casing: {self.ceramic_outer_radius*1000:.4f}mm to {self.reflective_outer_radius*1000:.4f}mm ({(self.reflective_outer_radius - self.ceramic_outer_radius)*1000:.4f} mm)")
            print(f"    - Optimized stretch factor: {config.REFLECTIVE_STRETCH_FACTOR}")
            print(f"    - Number of nodes: {self.reflective_num_nodes} (Compared to: {self.reflective_nodes_without_stretching}), Compression ratio: {self.reflective_compression_ratio:.4f}")
            print(f"    - Min-to-max ratio: {self.reflective_minmax_ratio:.4f}")
            print(f"    - Min spacing: {self.reflective_min_dr*1000:.4f} mm, Max spacing: {self.reflective_max_dr*1000:.4f} mm")
            print(f"    - Min diff spacing: {self.reflective_min_diff_dr*1000:.4f} mm, Max diff spacing: {self.reflective_max_diff_dr*1000:.4f} mm")
            print("----------------------------------------------------------")
            print("Summary: Excluded Sample and Kanthal Regions")
            print(f"Total number of nodes: {self.total_num_nodes}")
            print(f"Total number of nodes without hyperbolic stretching: {self.total_nodes_without_stretching}")
            print(f"Compression Ratio: {self.total_compression_ratio:.4f}")
        print("----------------------------------------------------------")
        print("Summary: Excluded Sample, Kanthal and Reflective Regions")
        print(f"Total number of nodes: {self.total_num_nodes_without_reflective}")
        print(f"without hyperbolic stretching: {self.total_nodes_without_stretching_without_reflective}")
        print(f"Compression Ratio: {self.total_compression_ratio_without_reflective:.4f}")
        if config.DEBUG_MODE:
            pd.DataFrame(self.glass_grid).to_csv(os.path.join(config.DEBUG_DIR, "hyperbolic_glass_grid.csv"), index=False)
            pd.DataFrame(self.cement_grid).to_csv(os.path.join(config.DEBUG_DIR, "hyperbolic_cement_grid.csv"), index=False)
            pd.DataFrame(self.ceramic_grid).to_csv(os.path.join(config.DEBUG_DIR, "hyperbolic_ceramic_grid.csv"), index=False)
            pd.DataFrame(self.reflective_grid).to_csv(os.path.join(config.DEBUG_DIR, "hyperbolic_reflective_grid.csv"), index=False)

if __name__ == "__main__":
    # Create an instance of the Test class
    self = HyperbolicGridGenerator()
    self.print_summary()

