#!/usr/bin/env python3
"""
COMPREHENSIVE TUBE FURNACE 3D VISUALIZATION SYSTEM
=================================================
Complete visualization suite that reads actual config, materials, mesh, and solver
to generate:
1. 3D Visualization of simulation setup layout
2. Cross-section and longitudinal views of the setup
3. High-resolution mesh structure and heat generation maps
4. HTML display with all information
5. Temperature profiles based on actual structure

Created: October 2025
Author: Tube Furnace Simulation Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Wedge
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
import json
import webbrowser

# Import all project modules
from config import *
from materials import TubeFurnaceMaterials, MATERIALS_DB
from mesh import TubeFurnaceMesh
from solver import TubeFurnaceHeatSolver
from mesh_3d_visualizer import *
from mesh_cross_section_analyzer import *
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("WARNING: Plotly not available - HTML output will be limited")

class ComprehensiveTubeFurnaceVisualization:
    """Complete 3D visualization system for tube furnace simulation"""
    
    def __init__(self):
        """Initialize with actual project components"""
        print("Initializing Comprehensive Tube Furnace Visualization...")
        
        # Initialize all components from actual project files
        self.materials = MATERIALS_DB
        self.mesh = TubeFurnaceMesh()
        self.solver = TubeFurnaceHeatSolver()
        
        # Generate mesh and setup
        self.mesh.generate_complete_mesh()
        
        # Initialize solver for temperature analysis
        self.solver.initialize_simulation()
        
        # Define material layer structure from actual config
        self.material_layers = self._extract_material_layers()
        
        # Define key positions for monitoring
        self.key_positions = self._define_key_positions()
        
        # Color scheme for materials
        self.material_colors = {
            'Sample': '#87CEEB',      # Sky blue
            'Glass': '#D3D3D3',       # Light gray
            'Cement': '#8B4513',      # Saddle brown
            'Ceramic': '#FFD700',     # Gold
            'Reflective': '#C0C0C0',  # Silver
            'Air Gap': '#E0FFFF',     # Light cyan
            'Al_5052': '#696969'      # Dim gray
        }
        
        print("Initialization complete!")
    
    def _extract_material_layers(self):
        """Extract material layer definitions from actual config"""
        layers = []
        
        # Layer 1: Sample space
        layers.append({
            'name': 'Sample',
            'r_inner': 0.0,
            'r_outer': GLASS_TUBE_INNER_RADIUS,
            'material_id': 0,
            'description': 'Sample space for materials under test'
        })
        
        # Layer 2: Quartz glass tube
        layers.append({
            'name': 'Quartz',
            'r_inner': GLASS_TUBE_INNER_RADIUS,
            'r_outer': GLASS_TUBE_OUTER_RADIUS,
            'material_id': 1,
            'description': 'Quartz glass containment tube'
        })
        
        # Layer 3: Imperial cement coating
        layers.append({
            'name': 'Cement',
            'r_inner': GLASS_TUBE_OUTER_RADIUS,
            'r_outer': GLASS_TUBE_OUTER_RADIUS + FURNACE_CEMENT_THICKNESS,
            'material_id': 2,
            'description': 'Imperial High Temperature Stove & Furnace Cement'
        })
        
        # Layer 4: Lyrufexon ceramic insulation
        layers.append({
            'name': 'Ceramic',
            'r_inner': GLASS_TUBE_OUTER_RADIUS + FURNACE_CEMENT_THICKNESS,
            'r_outer': INSULATION_OUTER_RADIUS,
            'material_id': 3,
            'description': 'Lyrufexon 101.8mm ceramic insulation'
        })
        
        # Layer 5: Reflective aluminum casing
        layers.append({
            'name': 'Reflective',
            'r_inner': INSULATION_OUTER_RADIUS,
            'r_outer': REFLECTIVE_CASING_OUTER_RADIUS,
            'material_id': 4,
            'description': 'Reflective aluminum casing'
        })
        
        # Layer 6: Air gap
        layers.append({
            'name': 'Air Gap',
            'r_inner': REFLECTIVE_CASING_OUTER_RADIUS,
            'r_outer': REFLECTIVE_CASING_OUTER_RADIUS + AIR_GAP_THICKNESS,
            'material_id': 5,
            'description': 'Air gap for thermal isolation'
        })
        
        # Layer 7: Aluminum 5052 cubic enclosure
        layers.append({
            'name': 'Al_5052',
            'r_inner': REFLECTIVE_CASING_OUTER_RADIUS + AIR_GAP_THICKNESS,
            'r_outer': REFLECTIVE_CASING_OUTER_RADIUS + AIR_GAP_THICKNESS + ENCLOSURE_WALL_THICKNESS,
            'material_id': 6,
            'description': 'Aluminum 5052 cubic enclosure (3mm wall)'
        })
        
        return layers
    
    def _define_key_positions(self):
        """Define key monitoring positions from actual config"""
        return {
            'cold_zone_front': 0.020,      # 20mm - front cold zone monitoring point
            'heating_start': HEATING_COIL_START,  # From config
            'heating_center': (HEATING_COIL_START + HEATING_COIL_END) / 2,
            'heating_end': HEATING_COIL_END,      # From config
            'cold_zone_back': FURNACE_LENGTH - 0.020 # 20mm from end - back cold zone monitoring point
        }

    def run_temperature_simulation(self):
        """Run short temperature simulation for visualization"""
        
        print(f"Running temperature simulation for {config.SIMULATION_DURATION/3600:.1f} hours...")
        
        # Run simulation
        self.solver.run_simulation()
        
        # Extract temperature data
        temperature_field = self.solver.T.copy()
        
        print(f"Simulation complete. Temperature range: {np.min(temperature_field)-273:.1f}°C to {np.max(temperature_field)-273:.1f}°C")
        
        return temperature_field
    
    def create_temperature_profile_analysis(self, temperature_field=None, save_path=None):
        """Create comprehensive temperature profile analysis"""
        
        if temperature_field is None:
            temperature_field = self.run_temperature_simulation()
        
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                            hspace=0.3, wspace=0.3)
        
        print("Creating temperature profile analysis...")
        
        # Convert temperature to Celsius
        temp_celsius = temperature_field - 273.15
        
        # === TOP LEFT: Axial Temperature Profile ===
        ax1 = fig.add_subplot(gs[0, 0])
        
        z_mm = self.mesh.z_nodes * 1000
        
        # Centerline temperature (r=0)
        centerline_temp = temp_celsius[0, :]
        
        # Sample space edge temperature
        sample_edge_idx = np.argmin(np.abs(self.mesh.r_nodes - GLASS_TUBE_INNER_RADIUS*0.9))
        sample_edge_temp = temp_celsius[sample_edge_idx, :]
        
        # Coil position temperature
        coil_idx = np.argmin(np.abs(self.mesh.r_nodes - HEATING_COIL_RADIUS))
        coil_temp = temp_celsius[coil_idx, :]
        
        ax1.plot(z_mm, centerline_temp, 'b-', linewidth=3, label='Centerline (r=0)')
        ax1.plot(z_mm, sample_edge_temp, 'g-', linewidth=2, label='Sample Edge')
        ax1.plot(z_mm, coil_temp, 'r-', linewidth=2, label='Coil Position')
        
        # Add heating zone markers
        ax1.axvline(HEATING_COIL_START*1000, color='orange', linestyle='--', alpha=0.7, label='Heating Start')
        ax1.axvline(HEATING_COIL_END*1000, color='orange', linestyle='--', alpha=0.7, label='Heating End')
        
        ax1.set_xlabel('Axial Position (mm)')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Axial Temperature Profiles')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # === TOP MIDDLE: Radial Temperature Profile ===
        ax2 = fig.add_subplot(gs[0, 1])
        
        r_mm = self.mesh.r_nodes * 1000
        
        # Temperature at heating center
        heating_center_idx = np.argmin(np.abs(self.mesh.z_nodes - (HEATING_COIL_START + HEATING_COIL_END)/2))
        radial_temp_center = temp_celsius[:, heating_center_idx]
        
        # Temperature at heating start
        heating_start_idx = np.argmin(np.abs(self.mesh.z_nodes - HEATING_COIL_START))
        radial_temp_start = temp_celsius[:, heating_start_idx]
        
        # Temperature outside heating zone
        cold_zone_idx = np.argmin(np.abs(self.mesh.z_nodes - 0.02))  # 20mm from inlet
        radial_temp_cold = temp_celsius[:, cold_zone_idx]
        
        ax2.plot(r_mm, radial_temp_center, 'r-', linewidth=3, label='Heating Center')
        ax2.plot(r_mm, radial_temp_start, 'orange', linewidth=2, label='Heating Start')
        ax2.plot(r_mm, radial_temp_cold, 'b-', linewidth=2, label='Cold Zone')
        
        # Add material boundaries
        for layer in self.material_layers:
            r_boundary = layer['r_outer'] * 1000
            if r_boundary < max(r_mm):
                ax2.axvline(r_boundary, color='gray', alpha=0.5, linestyle=':')
                ax2.text(r_boundary, max(radial_temp_center)*0.9, layer['name'], 
                        rotation=90, ha='right', va='top', fontsize=9)
        
        ax2.set_xlabel('Radial Position (mm)')
        ax2.set_ylabel('Temperature (°C)')
        ax2.set_title('Radial Temperature Profiles')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # === TOP RIGHT: 2D Temperature Field ===
        ax3 = fig.add_subplot(gs[0, 2])
        
        Z_mesh, R_mesh = np.meshgrid(z_mm, r_mm)
        im = ax3.contourf(Z_mesh, R_mesh, temp_celsius, levels=20, cmap='hot')
        plt.colorbar(im, ax=ax3, label='Temperature (°C)')
        
        # Add heating zone
        ax3.axvline(HEATING_COIL_START*1000, color='white', linestyle='--', linewidth=2)
        ax3.axvline(HEATING_COIL_END*1000, color='white', linestyle='--', linewidth=2)
        ax3.axhline(HEATING_COIL_RADIUS*1000, color='cyan', linestyle='-', linewidth=2)
        
        ax3.set_xlabel('Axial Position (mm)')
        ax3.set_ylabel('Radial Position (mm)')
        ax3.set_title('2D Temperature Field')
        
        # === BOTTOM: Temperature Statistics ===
        ax4 = fig.add_subplot(gs[1, :])
        
        # Create temperature distribution histogram
        temp_flat = temp_celsius.flatten()
        ax4.hist(temp_flat, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(np.mean(temp_flat), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {np.mean(temp_flat):.1f}°C')
        ax4.axvline(np.median(temp_flat), color='blue', linestyle='-', linewidth=2, 
                   label=f'Median: {np.median(temp_flat):.1f}°C')
        ax4.axvline(np.max(temp_flat), color='darkred', linestyle='--', linewidth=2, 
                   label=f'Max: {np.max(temp_flat):.1f}°C')
        ax4.axvline(np.min(temp_flat), color='darkblue', linestyle='--', linewidth=2, 
                   label=f'Min: {np.min(temp_flat):.1f}°C')
        
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Temperature Distribution Statistics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Statistics:\\n' + \
                    f'Mean: {np.mean(temp_flat):.1f}°C\\n' + \
                    f'Std Dev: {np.std(temp_flat):.1f}°C\\n' + \
                    f'Range: {np.max(temp_flat)-np.min(temp_flat):.1f}°C\\n' + \
                    f'Nodes > 100°C: {np.sum(temp_flat > 100)}/{len(temp_flat)} ({100*np.sum(temp_flat > 100)/len(temp_flat):.1f}%)'
        
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Temperature Profile Analysis - Tube Furnace Simulation', 
                    fontsize=16, fontweight='bold', y=0.96)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Temperature analysis saved: {save_path}")
        
        return fig
    
def main():
    """Main function to run comprehensive visualization suite"""
    print("COMPREHENSIVE TUBE FURNACE 3D VISUALIZATION SYSTEM")
    print("=" * 80)
    
    # Initialize system
    viz = ComprehensiveTubeFurnaceVisualization()
    
    # Create output directory
    output_dir = 'comprehensive_visualization_output'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\\nOutput directory: {output_dir}")
    
    # Generate all visualizations
    print("\\nGenerating all visualizations...")

    user_choice1 = input("\\n1. Run 3D Setup Visualization? (yes/no): ")    
    if user_choice1.lower() == 'yes':
        print("\\n1. Creating 3D Setup Visualization...")
        fig3D, ax3D = visualize_3d_mesh()
        fig3D.savefig(os.path.join(output_dir, '3d_setup.png'), dpi=300, bbox_inches='tight')

    user_choice2 = input("\\n2. Run Cross-Section View? (yes/no): ")
    if user_choice2.lower() == 'yes':
        print("\\n2. Creating Cross-Section View...")
        analyzer, figcrosssection = generate_detailed_cross_sections()
        figcrosssection.savefig(os.path.join(output_dir, 'cross_section.png'), dpi=300, bbox_inches='tight')

    print("\\n3. Creating Longitudinal View...")
    print("\\n3. Not Implement at the Moment")

    user_choice = input("\\n4. Run Temperature Simulation? (yes/no): ")
    # Check the user's answer
    if user_choice.lower() == 'yes':
        print("\\n4. Running Temperature Simulation...")
        temp_field = viz.run_temperature_simulation()
    
        print("\\n5. Creating Temperature Profile Analysis...")
        viz.create_temperature_profile_analysis(temp_field, 
                                           save_path=os.path.join(output_dir, 'temperature_analysis.png'))

    print("\\nCOMPREHENSIVE VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}/")
    
if __name__ == "__main__":
    main()