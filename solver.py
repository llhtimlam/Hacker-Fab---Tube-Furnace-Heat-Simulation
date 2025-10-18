"""
High-Resolution Heat Transfer Solver
Advanced finite difference solver with adaptive time stepping
"""

import numpy as np
import h5py  # Optional - only needed for data export
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
from functools import cached_property
import config
from materials import TubeFurnaceMaterials
from mesh import TubeFurnaceMesh

class TubeFurnaceHeatSolver:
    """Advanced heat transfer solver with high spatial and temporal resolution"""
    
    def __init__(self):
        self.mesh = TubeFurnaceMesh()
        self.materials = TubeFurnaceMaterials()
        
        # Grid dimensions 
        # | faces, o centers, < > Shifted -/+
        # for example
        # < | all faces shift left
        # most left duplicated (most left unusable and trimmed) (|)| 
        # most right got moved to left |()
        # vice versa
        self.num_r = len(self.mesh.r_centers) # centered r nodes |o|o|
        self.num_z = len(self.mesh.z_centers) # centered z nodes |o|o|

        self.r_centers = self.mesh.r_centers # 1D r o
        self.z_centers = self.mesh.z_centers # 1D z o
        self.r_faces = self.mesh.r_faces # 1D r+1 |
        self.z_faces = self.mesh.z_faces # 1D z+1 |

        # Create 2D grids of mesh properties
        self.dr_centers = np.diff(self.r_centers, prepend=self.r_centers[0], append=self.r_centers[-1]) # 1D r+1 |
        self.dz_centers = np.diff(self.z_centers, prepend=self.z_centers[0], append=self.z_centers[-1]) # 1D z+1 |

        # Precompute non-uniform grid spacings
        self.dr_grid = np.tile(np.diff(self.dr_centers), (self.num_z, 1)).T # 2D r*z  o
        self.dz_grid = np.tile(np.diff(self.dz_centers), (self.num_r, 1)) # 2D r*z  o
        # Negatively-biased face, Both faces got trimmed by 1, right centered got trimmed but left remained # |(0) o(0) |(1) and # |(n-1) o(n-1) |(n) o(n) extra
        self.dr1_grid = np.tile(np.diff(self.r_faces[1:-1] - self.r_centers[:-1], prepend=self.r_centers[0]), (self.num_z, 1)).T # 2D r-1*z <
        self.dr2_grid = np.tile(np.diff(self.r_centers[:-1] - self.r_faces[1:-1], append=self.r_faces[-1]), (self.num_z, 1)).T # 2D r-1*z >
        self.dz1_grid = np.tile(np.diff(self.z_faces[1:-1] - self.z_centers[:-1], prepend=self.z_centers[0]), (self.num_r, 1)) # 2D r*z-1 <
        self.dz2_grid = np.tile(np.diff(self.z_centers[:-1] - self.z_faces[1:-1], append=self.z_faces[-1]), (self.num_r, 1)) # 2D r*z-1 >
        
        # Precompute face areas and volumes
        self.volume, self.area_inner_face, self.area_outer_face, self.axial_face_area = self.precompute_face_volumes()
        self.cubic_outer_surface = self.mesh.cubic_outer_surface
        self.cubic_inner_surface = self.mesh.cubic_inner_surface
        self.air_gap_volume = self.mesh.cubic_air_gap_volume
        self.cubic_aluminium_casing_volume = self.mesh.cubic_aluminium_casing_volume
        # Settings
        self.ENABLE_RADIATION = config.ENABLE_RADIATION
        # Constants
        self.AMBIENT_TEMP = config.AMBIENT_TEMP
        self.h_conv_sample_air_space = config.H_CONV_SAMPLE_AIR_SPACE # 5 W/(m^2*K)
        self.h_conv_air_gap = config.H_CONV_AIR_GAP # 5 W/(m^2*K)
        self.h_conv_ambient = config.H_CONV_AMBIENT # 10 W/(m^2*K)
        self.STEFAN_BOLTZMANN = config.STEFAN_BOLTZMANN

        # Precompute material properties arrays for efficiency
        self.air_gap_mass = self.air_gap_volume * 1.225 # 0: 'Air' Density
        self.rho_aluminum_casing = self.materials.get_property(self.materials.material_index_to_name[7], "density", config.INITIAL_TEMP) # 7: 'aluminum_5052'
        self.aluminum_casing_mass = self.cubic_aluminium_casing_volume * self.rho_aluminum_casing
        self.emissivity_reflective = self.materials.get_emissivity(self.materials.material_index_to_name[5]) # 5: 'reflective_aluminum'
        self.emissivity_casing = self.materials.get_emissivity(self.materials.material_index_to_name[7]) # 7: 'aluminum_5052'
        
        # Initialized Lump Node Settings
        self.T_sample_air_space_avg = config.INITIAL_TEMP # Future implementation to N2 H2O(g) gas flow
        self.T_air_gap_avg = config.INITIAL_TEMP
        self.T_aluminum_casing_avg = config.INITIAL_TEMP
        self.sample_air_space_volume = self.mesh.sample_air_space_volume

        # Simulation state variables
        self.T = None  # Temperature field
        self.T_old = None  # Previous time step temperature
        self.dt = config.TIME_STEP_SECONDS  # Current time step
        self.time = 0.0  # Current simulation time
        self.setup_complete = False

        # Results storage
        self.temperature_history = []
        self.time_history = []
        self.key_point_history = {}
        
        # Performance monitoring
        self.solve_times = []
        self.total_iterations = 0
        
    def initialize_simulation(self):
        """Initialize mesh, materials, and temperature field"""
        print("Initializing high-resolution simulation...")
        # Generate mesh system

        self.mesh.generate_complete_mesh()
        self.material_map_cylindrical_centered = self.mesh.material_map_cylindrical_centered
        # Initialize temperature fields for Cylindrical region
        self.T = np.full((len(self.mesh.r_centers), len(self.mesh.z_centers)), config.INITIAL_TEMP)
        self.T_old = self.T.copy()
        
        self.net_heat_flow = np.zeros((self.num_r, self.num_z)) # 2D r*z o
        self.Q_gen_grid = self.precompute_heat_source_grid(config) # 2D r*z o
        # Precompute boundary condition grids
        self.boundary_info = self.boundary_data
        
        # Print initialization summary
        #self._validate_configuration()
        
        self.setup_complete = True
        
    def _validate_configuration(self):
        """Validate that hybrid mesh and solver configurations are properly matched"""
        print("\nMESH-SOLVER CONFIGURATION VALIDATION:")
        cylindrical_nodes = len(self.mesh.r_centers) * len(self.mesh.z_centers)
        total_nodes = cylindrical_nodes
        print(f"  Coil-node constraint: {config.HEATING_COIL_TURNS} coil turns = {config.AXIAL_NODES_HEATING} heating nodes (ENFORCED)")
        print(f"  Temperature fields initialized at {config.INITIAL_TEMP:.1f}K")
        print(f"  Coordinate systems: Cylindrical")
        print(f"  Cylindrical boundary: r = {self.mesh.cylindrical_outer_radius*1000:.1f}mm")
        print(f"  Cylindrical region: {len(self.mesh.r_centers)} × {len(self.mesh.z_centers)} = {cylindrical_nodes:,} nodes")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"Time steps: {config.TOTAL_TIME_STEPS:,} ({config.SIMULATION_DURATION/3600:.1f} hours)")
        print(f"Initial time step: {self.dt:.3f} seconds")
        print(f"Expected memory usage: ~{total_nodes * config.TOTAL_TIME_STEPS * 8 / 1e9:.1f} GB") 
    
    def convert_cylindrical_to_cartesian(self, r, theta, z):
        """Convert cylindrical coordinates to Cartesian"""
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y, z
    
    def convert_cartesian_to_cylindrical(self, x, y, z):
        """Convert Cartesian coordinates to cylindrical"""
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return r, theta, z
    
    def precompute_face_volumes(self):
        r_inner_face = np.tile(self.r_faces[:-1], (self.num_z, 1)).T  # 2D r*z < |
        r_outer_face = np.tile(self.r_faces[1:], (self.num_z, 1)).T # 2D r*z | >
        dz_grid = np.tile(np.diff(self.z_faces), (self.num_r, 1)) # 2D r*z o
        
        area_inner_face = 2 * np.pi * r_inner_face * dz_grid # 2D r*z < |
        area_outer_face = 2 * np.pi * r_outer_face * dz_grid # 2D r*z | >
        axial_face_area = np.pi * (r_outer_face**2 - r_inner_face**2)
        volume = np.pi * (r_outer_face**2 - r_inner_face**2) * dz_grid # 2D r*z o

        return volume, area_inner_face, area_outer_face, axial_face_area

    def harmonic_mean_nonuniform(self, k1, k2, dx1, dx2):
        # Method for harmonic mean, already vectorized
        denominator = k1 * dx1 + k2 * dx2
        return np.where(denominator != 0, (k1 * k2 * (dx1 + dx2)) / denominator, 0)

    def calculate_heat_source(self):      
        # Base volumetric heat generation
        pitch = config.HEATING_COIL_LENGTH / config.HEATING_COIL_TURNS
        wire_area = (np.pi * (config.HEATING_COIL_WIRE_DIAMETER / 2 ) **2 )
        wire_volume = wire_area * config.HEATING_COIL_TURNS * ((2 * np.pi * config.HEATING_COIL_RADIUS) ** 2 + pitch**2) ** 0.5
        
        normalized_volume = np.pi * (config.HEATING_COIL_RADIUS ** 2 - config.GLASS_TUBE_OUTER_RADIUS ** 2) * config.HEATING_COIL_LENGTH # Normalized Helical Wire to Average Heat Generated per Volume
        
        volumetric_heat_density = config.HEATING_COIL_POWER / wire_volume
        Q_dot_volumetric = config.HEATING_COIL_POWER / normalized_volume  # W/m (along coil length)
        if config.POWER_DENSITY_TREATMENT == "2D_CYLINDRICAL":
            return Q_dot_volumetric

    def precompute_heat_source_grid(self, config):
        # Create a boolean mask for the heating zone
        heating_zone_mask = self.material_map_cylindrical_centered == 2  # 2: 'kanthal_coil'
        Q_dot_volumetric = self.calculate_heat_source()
        # Create the heat generation grid, zero everywhere except the heating zone
        Q_gen_grid = np.zeros((self.num_r, self.num_z))
        Q_gen_grid[heating_zone_mask] = Q_dot_volumetric

        return Q_gen_grid
    
    def boundary_data(self):
        """
        Precomputes and caches all boundary masks and hA grids.
        Returns a dictionary with immutable (copied) NumPy arrays.
        """
        boundary_info = {
            "sample_glass": {
                "h": config.H_CONV_SAMPLE_AIR_SPACE
            },
            "reflective_airgap": {
                "h": config.H_CONV_AIR_GAP
            },
            "reflective_casing": {
                "h": self.emissivity_reflective * config.STEFAN_BOLTZMANN
            }
        }
        
        boundary_results = {}
        for name, data in boundary_info.items():
            mask_2d = np.zeros((self.num_r, self.num_z), dtype=bool)
            
            if name == "sample_glass":
                mask_2d[0, :] = True
                area_values = self.area_inner_face[0, :]
            elif name == "reflective_airgap" or name == "reflective_casing":
                mask_2d[self.num_r - 1, :] = True
                area_values = self.area_outer_face[self.num_r - 1, :]
                        
            # Calculate hA_grid using the 2D boolean mask
            hA_grid = np.zeros((self.num_r, self.num_z))
            hA_grid[mask_2d] = data["h"] * area_values
            
            # Store copies of the results to ensure immutability
            boundary_results[name] = {
                "mask": mask_2d.copy(),
                "hA_grid": hA_grid.copy()
            }
        return boundary_results
    
    def solve_timestep(self):
        # 1. Precompute temperature-dependent material properties
        # Cylindrical Continuous Conduction Region Material Properties Update
        k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.material_map_cylindrical_centered,
            self.T
        )
        # Lump Node Material Properties Update
        # Cylindrical Sample Air Space Convection Region
        _, _, cp_sample_air_space = self.materials.get_thermal_properties(
        self.materials.material_index_to_name.get(0), self.T_sample_air_space_avg)
        cv_sample_air_space = cp_sample_air_space - 287.05 # J/(kg·K) Ideal gas constant for air
        sample_air_space_mass = self.sample_air_space_volume * 1.225 # kg/m^3 at room temp, not dealing with dH
        # Cubic Convection and Radiation Region
        _, _, cp_air_gap = self.materials.get_thermal_properties(
        self.materials.material_index_to_name.get(6), self.T_air_gap_avg) # 6: 'Air'
        _, _, cp_aluminum_casing = self.materials.get_thermal_properties(
        self.materials.material_index_to_name.get(7), self.T_aluminum_casing_avg)
        cv_air_gap = cp_air_gap - 287.05  # J/(kg·K) Ideal gas constant for air

        # 2. Store pre-calculated harmonic mean conductivities
        k_r_grid = self.harmonic_mean_nonuniform(k_matrix[:-1, :], k_matrix[1:, :], self.dr1_grid, self.dr2_grid) # 2D r-1*z o(|)o
        k_z_grid = self.harmonic_mean_nonuniform(k_matrix[:, :-1], k_matrix[:, 1:], self.dz1_grid, self.dz2_grid) # 2D r*z-1 o(|)o
        
        # 3. Precompute vectorized fluxes for internal cell for Finite Volume Method (FVM)
        #dT_dr_grid = (self.T[1:, :] - self.T[:-1, :]) / self.dr_minus_grid[1:, :] # 2D r-1*z o(|)o # trimmed <(|)| duplicated unusable 0
        #dT_dz_grid = (self.T[:, :-1] - self.T[:, 1:]) / self.dz_minus_grid[:, 1:] # 2D r*z-1 o(|)o # trimmed <(|)| duplicated unusable 0
        radial_flux_minus = k_r_grid * self.area_inner_face[1:, :] * (self.T[1:, :] - self.T[:-1, :]) / self.dr_grid[:-1, :] # 2D r-1*z < | Ti - Ti-1
        radial_flux_plus = k_r_grid * self.area_outer_face[:-1, :] * (self.T[1:, :] - self.T[:-1, :]) / self.dr_grid[1:, :] # 2D r-1*z | > Ti+1 - Ti
        axial_flux_minus = k_z_grid * self.axial_face_area[:, 1:] * (self.T[:, 1:] - self.T[:, :-1]) / self.dz_grid[:, -1:] # 2D r*z-1 < | Ti - Ti-1
        axial_flux_plus = k_z_grid * self.axial_face_area[:, :-1] * (self.T[:, 1:] - self.T[:, :-1]) / self.dz_grid[:, :1] # 2D r*z-1 | > Ti+1 - Ti

        # 4. Handle Heat source and sink
        
        boundaries = self.boundary_data()
        hA_conv_to_sample_air_space_local = boundaries["sample_glass"]["hA_grid"]
        hA_conv_to_air_gap_local = boundaries["reflective_airgap"]["hA_grid"]
        hA_rad_out_reflective_grid = boundaries["reflective_casing"]["hA_grid"]

        Q_conv_to_sample_air_space_local = hA_conv_to_sample_air_space_local * (self.T - self.T_sample_air_space_avg)
        Q_conv_to_air_gap_local = hA_conv_to_air_gap_local * (self.T - self.T_air_gap_avg)
        Q_rad_out_reflective_local = hA_rad_out_reflective_grid * (self.T**4 - self.T_aluminum_casing_avg**4)

        Q_conv_to_sample_air_space_total = Q_conv_to_sample_air_space_local.sum()
        Q_conv_to_air_gap_total = Q_conv_to_air_gap_local.sum()
        Q_rad_out_reflective_total = Q_rad_out_reflective_local.sum()

        Q_conv_from_air_gap_total = self.h_conv_air_gap * self.cubic_inner_surface * (self.T_air_gap_avg - self.T_aluminum_casing_avg)
        
        Q_conv_to_ambient = self.h_conv_ambient * self.cubic_outer_surface * (self.T_aluminum_casing_avg - config.AMBIENT_TEMP)
        Q_rad_to_ambient = self.emissivity_casing * config.STEFAN_BOLTZMANN * self.cubic_outer_surface * (self.T_aluminum_casing_avg**4 - config.AMBIENT_TEMP**4)
        
        # 5. Assemble net heat flow and update temperature field
        self.net_heat_flow[1:-1, 1:-1] = (radial_flux_minus[1:, 1:-1] - radial_flux_plus[:-1, 1:-1] + 
                                        axial_flux_minus[1:-1, 1:] - axial_flux_plus[1:-1, :-1])
        self.net_heat_flow += self.Q_gen_grid * self.volume
        self.net_heat_flow -= Q_conv_to_sample_air_space_local
        self.net_heat_flow -= Q_conv_to_air_gap_local + Q_rad_out_reflective_local
        dT_dt_grid = self.net_heat_flow / (rho_matrix * cp_matrix * self.volume)
        
        dT_dt_sample_air_space_avg = Q_conv_to_sample_air_space_total / (sample_air_space_mass * cv_sample_air_space)
        dT_dt_air_gap = (Q_conv_to_air_gap_total - Q_conv_from_air_gap_total) / (self.air_gap_mass * cv_air_gap)
        dT_dt_to_aluminum_casing = (Q_conv_from_air_gap_total - Q_conv_to_ambient + Q_rad_out_reflective_total - Q_rad_to_ambient) / (self.aluminum_casing_mass * cp_aluminum_casing)
        
        self.T_sample_air_space_avg += dT_dt_sample_air_space_avg * self.dt
        self.T_air_gap_avg += dT_dt_air_gap * self.dt
        self.T_aluminum_casing_avg += dT_dt_to_aluminum_casing * self.dt
    
        # Update cylindrical temperature field
        
        self.T += dT_dt_grid * self.dt
    
    def solve_heat_equation_hybrid(self):
        """Solve heat equation for Finite Volume Method (FVM) Lumped-Element Model Hybrid Heat Simulation System"""
        start_time = time.time()
        self.solve_timestep()
        # Record solve time
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)
        return solve_time
    
    def calculate_time_step(self):
        """Calculate adaptive time step based on CFL condition"""
        # Calculate minimum mesh spacing
        dr_min = np.min(np.diff(self.mesh.r_nodes))
        dz_min = np.min(np.diff(self.mesh.z_nodes))
        dx_min = min(dr_min, dz_min)
        k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.material_map_cylindrical_centered,
            self.T
        )
        # Maximum thermal diffusivity in domain
        alpha_max = 0.0
        for i in range(0, len(self.mesh.r_nodes), 3):  # Sample every 3rd node for efficiency
            for j in range(0, len(self.mesh.z_nodes), 3):  # Sample every 3rd node for efficiencyj]
                alpha = k_matrix[i, j] / (rho_matrix[i, j] * cp_matrix[i, j])
                alpha_max = max(alpha_max, alpha)
        
        # CFL condition: dt ≤ dx²/(2α) for 2D
        dt_cfl = config.CFL_SAFETY_FACTOR * dx_min**2 / (4 * alpha_max)
        
        # Apply time step limits
        dt_new = np.clip(dt_cfl, config.MIN_TIME_STEP, config.MAX_TIME_STEP)
        return dt_new
        
    def run_simulation(self):
        """Run complete high-resolution simulation"""
        if not self.setup_complete:
            self.initialize_simulation()
        
        print(f"\nRunning HIGH-RESOLUTION simulation...")
        print(f"Duration: {config.SIMULATION_DURATION/3600:.1f} hours")
        print(f"Time steps: {config.TOTAL_TIME_STEPS:,}")
        print(f"Initial time step: {self.dt:.3f} seconds")
        
        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        
        # Initialize HDF5 file for large dataset storage
        h5_filename = os.path.join(config.OUTPUT_DIR, config.TEMPERATURE_OUTPUT_FILE)
        
        with h5py.File(h5_filename, 'w') as h5f:
            # Create datasets
            temp_dataset = h5f.create_dataset('temperature', 
                                            (config.TOTAL_TIME_STEPS, len(self.r_centers), len(self.z_centers)),
                                            dtype=np.float32, compression='gzip')
            time_dataset = h5f.create_dataset('time', (config.TOTAL_TIME_STEPS,), dtype=np.float32)
            lumped_temp_dataset = h5f.create_dataset('lumped_temperatures', (config.TOTAL_TIME_STEPS, 3), dtype=np.float32, compression='gzip') 
            # Store mesh information
            h5f.create_dataset('r_nodes', data=self.r_centers)
            h5f.create_dataset('z_nodes', data=self.z_centers)
            h5f.create_dataset('material_map', data=self.material_map_cylindrical_centered)

            # Time stepping loop
            step = 0
            with tqdm(total=config.TOTAL_TIME_STEPS, desc="Time steps", unit="step") as pbar:
                while step < config.TOTAL_TIME_STEPS and self.time < config.SIMULATION_DURATION:
                    # Adaptive time stepping
                    #if step % 100 == 0:  # Recalculate every 100 steps # Turn off assume user input right dt
                        #self.dt = self.calculate_time_step()
                    
                    # Solve heat equation
                    solve_time = self.solve_heat_equation_hybrid()
                    
                    # Update time
                    self.time += self.dt
                    if step % 1000 == 0:
                        # Record data
                        temp_dataset[step] = self.T.astype(np.float32)
                        time_dataset[step] = self.time
                        self.time_history.append(self.time)
                        lumped_temp_dataset[step] = self.T_sample_air_space_avg.astype(np.float32), self.T_air_gap_avg.astype(np.float32), self.T_aluminum_casing_avg.astype(np.float32)

                    # Update progress
                    pbar.set_postfix({
                        'Time': f'{self.time/3600:.2f}h',
                        'dt': f'{self.dt:.16f}s',
                        'T_max': f'{np.max(self.T)-273:.0f}°C',
                        'Solve': f'{solve_time:.3f}s'
                    })
                    pbar.update(1)
                    
                    step += 1
                    self.total_iterations += 1
        
        print(f"\n✅ High-resolution simulation completed!")
        print(f"Total time steps: {step:,}")
        print(f"Final time: {self.time/3600:.2f} hours")
        print(f"Average solve time: {np.mean(self.solve_times):.4f} seconds/step")
        print(f"Results saved to: {h5_filename}")
        
        return h5_filename
    
    def create_visualization(self, save_plots=True):
        """Create comprehensive visualization of results"""
        if len(self.time_history) == 0:
            print("No simulation data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Convert time to hours
        time_hours = np.array(self.time_history) / 3600
        
        # 2. Current temperature field
        ax2 = axes[0, 1]
        temp_plot = self.T - 273  # Convert to Celsius
        im2 = ax2.contourf(self.mesh.Z * 1000, self.mesh.R * 1000, temp_plot, 
                          levels=50, cmap='hot')
        plt.colorbar(im2, ax=ax2, label='Temperature (°C)')
        ax2.set_xlabel('Axial Position (mm)')
        ax2.set_ylabel('Radial Position (mm)')
        ax2.set_title(f'Temperature Field at t = {self.time/3600:.2f}h')
        
        # 3. Radial temperature profile at center
        ax3 = axes[0, 2]
        center_z_idx = len(self.mesh.z_nodes) // 2
        radial_profile = self.T[:, center_z_idx] - 273
        ax3.plot(self.mesh.r_nodes * 1000, radial_profile, 'r-', linewidth=3)
        ax3.set_xlabel('Radial Position (mm)')
        ax3.set_ylabel('Temperature (°C)')
        ax3.set_title('Radial Temperature Profile (Center)')
        ax3.grid(True, alpha=0.3)
        
        
        # 4. Axial temperature profile at heating coil
        ax4 = axes[1, 0]
        coil_r_idx = np.argmin(np.abs(self.mesh.r_nodes - config.HEATING_COIL_RADIUS))
        axial_profile = self.T[coil_r_idx, :] - 273
        ax4.plot(self.mesh.z_nodes * 1000, axial_profile, 'b-', linewidth=3)
        ax4.set_xlabel('Axial Position (mm)')
        ax4.set_ylabel('Temperature (°C)')
        ax4.set_title('Axial Temperature Profile (Heating Coil)')
        ax4.grid(True, alpha=0.3)
        
        # 6. Simulation performance
        ax6 = axes[1, 2]
        if len(self.solve_times) > 0:
            ax6.plot(self.solve_times, 'g-', alpha=0.7)
            ax6.set_xlabel('Time Step')
            ax6.set_ylabel('Solve Time (seconds)')
            ax6.set_title('Solver Performance')
            ax6.grid(True, alpha=0.3)
            
            # Add statistics
            mean_time = np.mean(self.solve_times)
            ax6.axhline(mean_time, color='red', linestyle='--', 
                       label=f'Mean: {mean_time:.4f}s')
            ax6.legend()
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
            plt.savefig(os.path.join(config.VISUALIZATION_DIR, 'high_resolution_results.pdf'), 
                       dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(config.VISUALIZATION_DIR, 'high_resolution_results.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {config.VISUALIZATION_DIR}/")
        
        return fig
    

if __name__ == "__main__":

    # Run tube furnace simulation
    solver = TubeFurnaceHeatSolver()

    # Run simulation
    h5_file = solver.run_simulation()
    
    # Create visualizations
    solver.create_visualization()
    
    plt.show()
