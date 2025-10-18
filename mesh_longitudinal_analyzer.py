"""
Mesh Longitudinal Analyzer
Generates detailed longitudinal views of the tube furnace mesh
Shows axial node distribution and heating zones
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as patches
import config
from mesh import TubeFurnaceMesh

class MeshLongitudinalAnalyzer:
    """Analyze and visualize mesh longitudinal sections with node distribution"""
    
    def __init__(self):
        self.mesh = TubeFurnaceMesh()
        self.mesh.generate_complete_mesh()
        
        # Material colors (consistent with other visualization files)
        self.material_colors = {
            'sample_space': '#87CEEB',      # Sky blue
            'quartz_glass': '#D3D3D3', # Light gray
            'imperial_cement': '#8B4513',    # Saddle brown
            'lyrufexon_ceramic': '#FFD700',  # Gold
            'reflective_aluminum': '#C0C0C0', # Silver
            'air_gap': '#E0FFFF',           # Light cyan
            'aluminum_5052': '#696969',     # Dim gray
            'heating_coil': '#FF0000'       # Red
        }
        
    def generate_longitudinal_views(self):
        """Generate multiple longitudinal view analyses"""
        
        print("Generating longitudinal mesh analysis...")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Full longitudinal cross-section (top view)
        ax1 = plt.subplot(3, 2, 1)
        self._plot_full_longitudinal_section(ax1, view='top')
        
        # 2. Full longitudinal cross-section (side view)
        ax2 = plt.subplot(3, 2, 2)
        self._plot_full_longitudinal_section(ax2, view='side')
        
        # 3. Heating zone detail
        ax3 = plt.subplot(3, 2, 3)
        self._plot_heating_zone_detail(ax3)
        
        # 4. Node distribution analysis
        ax4 = plt.subplot(3, 2, 4)
        self._plot_node_density_analysis(ax4)
        
        # 5. Temperature profile zones
        ax5 = plt.subplot(3, 2, 5)
        self._plot_temperature_zones(ax5)
        
        # 6. Information panel
        ax6 = plt.subplot(3, 2, 6)
        self._add_longitudinal_info(ax6)
        
        plt.suptitle('Tube Furnace Longitudinal Analysis with Node Distribution', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_full_longitudinal_section(self, ax, view='top'):
        """Plot full longitudinal section with all material boundaries"""
        
        z = self.mesh.z_nodes
        
        if view == 'top':
            # Top view: show sample space to ceramic outer radius
            boundaries = [
                (self.mesh.sample_radius, 'Sample Space', self.material_colors['sample_space']),
                (self.mesh.glass_outer_radius, 'Glass Tube', self.material_colors['quartz_glass']),
                (self.mesh.cement_outer_radius, 'Cement + Coil', self.material_colors['imperial_cement']),
                (self.mesh.ceramic_outer_radius, 'Ceramic Wool', self.material_colors['lyrufexon_ceramic']),
                (self.mesh.reflective_outer_radius, 'Reflective Al', self.material_colors['reflective_aluminum'])
            ]
            
            for i, (radius, label, color) in enumerate(boundaries):
                ax.fill_between(z, -radius, radius, alpha=0.7, color=color, label=label)
            
            ax.set_ylim([-self.mesh.reflective_outer_radius * 1.1, 
                        self.mesh.reflective_outer_radius * 1.1])
            ax.set_ylabel('Radial Distance (m)')
            ax.set_title('Longitudinal Section - Cylindrical Region (Top View)', fontweight='bold')
            
        else:  # side view
            # Side view: show cubic region
            air_gap = self.mesh.air_gap_outer
            box_outer = self.mesh.aluminum_box_outer
            
            # Draw air gap region
            ax.fill_between(z, -air_gap, air_gap, alpha=0.5, 
                           color=self.material_colors['air_gap'], label='Air Gap')
            
            # Draw aluminum box walls (top and bottom)
            ax.fill_between(z, air_gap, box_outer, alpha=0.8, 
                           color=self.material_colors['aluminum_5052'], label='Aluminum Box')
            ax.fill_between(z, -box_outer, -air_gap, alpha=0.8, 
                           color=self.material_colors['aluminum_5052'])
            
            ax.set_ylim([-box_outer * 1.1, box_outer * 1.1])
            ax.set_ylabel('Y Distance (m)')
            ax.set_title('Longitudinal Section - Cubic Region (Side View)', fontweight='bold')
        
        # Plot heating coil region
        self._add_heating_zone_overlay(ax)
        
        # Plot node distribution
        self._add_node_markers(ax)
        
        ax.set_xlabel('Axial Position (m)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    def _add_heating_zone_overlay(self, ax):
        """Add heating zone overlay to longitudinal plots"""
        
        # Heating zone background
        heating_rect = Rectangle((config.HEATING_COIL_START, ax.get_ylim()[0]), 
                                config.HEATING_COIL_LENGTH, 
                                ax.get_ylim()[1] - ax.get_ylim()[0],
                                alpha=0.2, facecolor='red', edgecolor='darkred', 
                                linewidth=2, linestyle='--', label='Heating Zone')
        ax.add_patch(heating_rect)
        
        # Add zone labels
        ax.axvline(config.HEATING_COIL_START, color='red', linestyle='--', alpha=0.8)
        ax.axvline(config.HEATING_COIL_END, color='red', linestyle='--', alpha=0.8)
        
        ax.text(config.HEATING_COIL_START/2, ax.get_ylim()[1]*0.8, 
               'COLD\nZONE', ha='center', va='center', fontsize=10, 
               fontweight='bold', color='blue')
        
        ax.text((config.HEATING_COIL_START + config.HEATING_COIL_END)/2, 
               ax.get_ylim()[1]*0.8, 
               'HEATING\nZONE', ha='center', va='center', fontsize=10, 
               fontweight='bold', color='red')
        
        ax.text((config.HEATING_COIL_END + config.FURNACE_LENGTH)/2, 
               ax.get_ylim()[1]*0.8, 
               'COLD\nZONE', ha='center', va='center', fontsize=10, 
               fontweight='bold', color='blue')
    
    def _add_node_markers(self, ax):
        """Add node distribution markers to longitudinal view"""
        
        # Sample z positions to show node distribution
        z_sample_step = max(1, len(self.mesh.z_nodes) // 50)  # Show ~50 positions
        
        for i in range(0, len(self.mesh.z_nodes), z_sample_step):
            z_pos = self.mesh.z_nodes[i]
            
            # Add vertical lines to show mesh positions
            ax.axvline(z_pos, color='black', alpha=0.3, linewidth=0.5)
            
            # Add small markers at key radial positions
            if i % (z_sample_step * 3) == 0:  # Every 3rd z position
                ax.scatter(z_pos, 0, c='red', s=15, alpha=0.8, zorder=10)
    
    def _plot_heating_zone_detail(self, ax):
        """Plot detailed view of heating zone with coil representation"""
        
        # Focus on heating zone
        z_heating = np.linspace(config.HEATING_COIL_START, config.HEATING_COIL_END, 200)
        
        # Draw heating coil path
        coil_radius = config.HEATING_COIL_RADIUS
        turns_per_length = config.HEATING_COIL_TURNS / config.HEATING_COIL_LENGTH
        
        # Calculate coil position
        coil_angle = 2 * np.pi * turns_per_length * (z_heating - config.HEATING_COIL_START)
        coil_x = coil_radius * np.cos(coil_angle)
        coil_y = coil_radius * np.sin(coil_angle)
        
        # Plot coil path (top view)
        ax.plot(z_heating, coil_x, color='red', linewidth=3, alpha=0.8, label='Heating Coil Path')
        
        # Add material boundaries in heating zone
        boundaries = [
            (self.mesh.sample_radius, 'Sample'),
            (self.mesh.glass_outer_radius, 'Glass'),
            (self.mesh.cement_outer_radius, 'Cement'),
            (self.mesh.ceramic_outer_radius, 'Ceramic')
        ]
        
        for radius, label in boundaries:
            ax.axhline(radius, color='gray', alpha=0.5, linestyle='-', linewidth=1)
            ax.axhline(-radius, color='gray', alpha=0.5, linestyle='-', linewidth=1)
            ax.text(config.HEATING_COIL_START, radius, label, fontsize=8, 
                   verticalalignment='bottom')
        
        # Heating zone nodes
        z_indices = np.where((self.mesh.z_nodes >= config.HEATING_COIL_START) & 
                            (self.mesh.z_nodes <= config.HEATING_COIL_END))[0]
        
        for i in range(0, len(z_indices), max(1, len(z_indices)//20)):
            idx = z_indices[i]
            z_pos = self.mesh.z_nodes[idx]
            ax.axvline(z_pos, color='orange', alpha=0.6, linewidth=1)
        
        ax.set_xlim([config.HEATING_COIL_START - 0.01, config.HEATING_COIL_END + 0.01])
        ax.set_ylim([-self.mesh.ceramic_outer_radius * 1.2, 
                    self.mesh.ceramic_outer_radius * 1.2])
        ax.set_xlabel('Axial Position (m)')
        ax.set_ylabel('Radial Position (m)')
        ax.set_title('Heating Zone Detail with Coil Path', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_node_density_analysis(self, ax):
        """Plot node density distribution along furnace length"""
        
        z_positions = self.mesh.z_nodes
        
        # Calculate node density (nodes per unit length)
        z_spacing = np.diff(z_positions)
        z_centers = (z_positions[:-1] + z_positions[1:]) / 2
        node_density = 1.0 / z_spacing  # nodes per meter
        
        # Plot density
        ax.plot(z_centers, node_density, 'b-', linewidth=2, label='Axial Node Density')
        
        # Add heating zone overlay
        ax.axvspan(config.HEATING_COIL_START, config.HEATING_COIL_END, 
                  alpha=0.2, color='red', label='Heating Zone')
        
        # Add statistics
        avg_density = np.mean(node_density)
        heating_indices = np.where((z_centers >= config.HEATING_COIL_START) & 
                                  (z_centers <= config.HEATING_COIL_END))[0]
        
        if len(heating_indices) > 0:
            heating_density = np.mean(node_density[heating_indices])
            ax.axhline(heating_density, color='red', linestyle='--', alpha=0.8,
                      label=f'Heating Zone Avg: {heating_density:.1f} nodes/m')
        
        ax.axhline(avg_density, color='black', linestyle='--', alpha=0.8,
                  label=f'Overall Avg: {avg_density:.1f} nodes/m')
        
        ax.set_xlabel('Axial Position (m)')
        ax.set_ylabel('Node Density (nodes/m)')
        ax.set_title('Axial Node Density Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_temperature_zones(self, ax):
        """Plot temperature zones and boundary conditions"""
        
        z = self.mesh.z_nodes
        
        # Create temperature profile approximation
        temp_profile = np.zeros_like(z)
        
        for i, z_pos in enumerate(z):
            if z_pos < config.HEATING_COIL_START:
                # Cold zone - linear increase toward heating
                temp_profile[i] = config.AMBIENT_TEMP + \
                    (config.TARGET_TEMP - config.AMBIENT_TEMP) * \
                    (z_pos / config.HEATING_COIL_START) * 0.2
            elif z_pos <= config.HEATING_COIL_END:
                # Heating zone - high temperature
                temp_profile[i] = config.TARGET_TEMP
            else:
                # Cold zone - exponential decay
                decay_length = 0.1  # 10cm decay length
                temp_profile[i] = config.AMBIENT_TEMP + \
                    (config.TARGET_TEMP - config.AMBIENT_TEMP) * \
                    np.exp(-(z_pos - config.HEATING_COIL_END) / decay_length)
        
        # Plot temperature profile
        ax.plot(z, temp_profile, 'r-', linewidth=3, label='Temperature Profile')
        
        # Add zone boundaries
        ax.axvline(config.HEATING_COIL_START, color='orange', linestyle='--', 
                  linewidth=2, label='Heating Zone Start')
        ax.axvline(config.HEATING_COIL_END, color='orange', linestyle='--', 
                  linewidth=2, label='Heating Zone End')
        
        # Add temperature zones
        ax.axhspan(config.AMBIENT_TEMP - 50, config.AMBIENT_TEMP + 50, 
                  alpha=0.2, color='blue', label='Cold Zone Range')
        ax.axhspan(config.TARGET_TEMP - 100, config.TARGET_TEMP + 100, 
                  alpha=0.2, color='red', label='Heating Zone Range')
        
        # Boundary conditions
        ax.scatter([0, config.FURNACE_LENGTH], 
                  [config.AMBIENT_TEMP, config.AMBIENT_TEMP],
                  c='blue', s=100, marker='s', label='Cold BC', zorder=10)
        
        ax.set_xlabel('Axial Position (m)')
        ax.set_ylabel('Temperature (K)')
        ax.set_title('Temperature Zones and Boundary Conditions', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _add_longitudinal_info(self, ax):
        """Add information panel for longitudinal analysis"""
        
        ax.axis('off')
        
        # Title
        ax.text(0.02, 0.95, 'LONGITUDINAL ANALYSIS INFO', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        y_pos = 0.85
        line_spacing = 0.06
        
        # Mesh statistics
        ax.text(0.02, y_pos, 'MESH STATISTICS:', fontsize=12, fontweight='bold',
                transform=ax.transAxes, color='darkblue')
        y_pos -= line_spacing * 0.8
        
        total_nodes = len(self.mesh.r_nodes) * len(self.mesh.z_nodes)
        if hasattr(self.mesh, 'cubic_nodes_xyz'):
            total_nodes += len(self.mesh.cubic_nodes_xyz)
        
        stats = [
            f'Total axial positions: {len(self.mesh.z_nodes)}',
            f'Cylindrical nodes: {len(self.mesh.r_nodes) * len(self.mesh.z_nodes)}',
            f'Cubic nodes: {len(self.mesh.cubic_nodes_xyz) if hasattr(self.mesh, "cubic_nodes_xyz") else 0}',
            f'Total mesh nodes: {total_nodes}',
            f'Furnace length: {config.FURNACE_LENGTH*1000:.0f}mm',
            f'Heating zone length: {config.HEATING_COIL_LENGTH*1000:.0f}mm'
        ]
        
        for stat in stats:
            ax.text(0.05, y_pos, f'• {stat}', fontsize=10,
                   transform=ax.transAxes)
            y_pos -= line_spacing * 0.7
        
        y_pos -= line_spacing * 0.5
        
        # Heating zone details
        ax.text(0.02, y_pos, 'HEATING ZONE DETAILS:', fontsize=12, fontweight='bold',
                transform=ax.transAxes, color='darkred')
        y_pos -= line_spacing * 0.8
        
        heating_details = [
            f'Coil radius: {config.HEATING_COIL_RADIUS*1000:.1f}mm',
            f'Coil turns: {config.HEATING_COIL_TURNS}',
            f'Wire diameter: {config.HEATING_COIL_WIRE_DIAMETER*1000:.1f}mm',
            f'Start position: {config.HEATING_COIL_START*1000:.0f}mm',
            f'End position: {config.HEATING_COIL_END*1000:.0f}mm',
            f'Power: {config.HEATING_COIL_POWER:.0f}W'
        ]
        
        for detail in heating_details:
            ax.text(0.05, y_pos, f'• {detail}', fontsize=10,
                   transform=ax.transAxes)
            y_pos -= line_spacing * 0.7
        
        y_pos -= line_spacing * 0.5
        
        # Temperature conditions
        ax.text(0.02, y_pos, 'TEMPERATURE CONDITIONS:', fontsize=12, fontweight='bold',
                transform=ax.transAxes, color='purple')
        y_pos -= line_spacing * 0.8
        
        temp_conditions = [
            f'Cold zone temp: {config.AMBIENT_TEMP:.0f}K ({config.AMBIENT_TEMP-273:.0f}°C)',
            f'Heating zone temp: {config.TARGET_TEMP:.0f}K ({config.TARGET_TEMP-273:.0f}°C)',
            f'Temperature rise: {config.TARGET_TEMP-config.AMBIENT_TEMP:.0f}K'
        ]
        
        for condition in temp_conditions:
            ax.text(0.05, y_pos, f'• {condition}', fontsize=10,
                   transform=ax.transAxes)
            y_pos -= line_spacing * 0.7

def generate_detailed_longitudinal_analysis():
    """Main function to generate detailed longitudinal analysis"""
    
    print("Starting detailed longitudinal analysis...")
    
    analyzer = MeshLongitudinalAnalyzer()
    fig = analyzer.generate_longitudinal_views()
    
    print("Longitudinal analysis complete!")
    return analyzer, fig

if __name__ == "__main__":
    analyzer, fig = generate_detailed_longitudinal_analysis()
    print("Longitudinal visualization complete. Close the plot to continue.")