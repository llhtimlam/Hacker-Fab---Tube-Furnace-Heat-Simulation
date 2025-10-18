"""
Mesh Cross-Section Analyzer
Generates detailed cross-sectional views of the tube furnace mesh
Shows node distribution and material boundaries
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
import config
from mesh import TubeFurnaceMesh

class MeshCrossSectionAnalyzer:
    """Analyze and visualize mesh cross-sections with node distribution"""
    
    def __init__(self):
        self.mesh = TubeFurnaceMesh()
        self.mesh.generate_complete_mesh()
        
        # Create mesh data structure from cubic nodes
        self._prepare_mesh_data()
        
        # Material colors (consistent with cross_section_generator.py)
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
        
    def _prepare_mesh_data(self):
        """Prepare mesh data structure for analysis"""
        if hasattr(self.mesh, 'cubic_nodes_xyz') and self.mesh.cubic_nodes_xyz is not None:
            # Convert cubic nodes to structured data
            cubic_array = np.array(self.mesh.cubic_nodes_xyz)
            
            self.mesh_data = {
                'x': cubic_array[:, 0],  # X coordinates
                'y': cubic_array[:, 1],  # Y coordinates  
                'z': cubic_array[:, 2],  # Z coordinates
                'material': []  # Will be filled based on material mapping
            }
            
            # Create material labels based on material_map_cubic_sparse
            if hasattr(self.mesh, 'material_map_cubic_sparse'):
                materials = []
                for i, mat_code in enumerate(self.mesh.material_map_cubic_sparse):
                    if mat_code == 5:  # Air gap (from mesh.py line 295)
                        materials.append('air_gap')
                    elif mat_code == 6:  # Aluminum 5052 box (from mesh.py line 293)
                        materials.append('aluminum_5052')
                    else:
                        materials.append('unknown')
                self.mesh_data['material'] = np.array(materials)
            else:
                # Fallback: classify based on position
                materials = []
                for x, y, z in self.mesh.cubic_nodes_xyz:
                    if abs(x) > self.mesh.air_gap_outer or abs(y) > self.mesh.air_gap_outer:
                        materials.append('aluminum_5052')
                    else:
                        materials.append('air_gap')
                self.mesh_data['material'] = np.array(materials)
        else:
            # No cubic mesh data available
            self.mesh_data = None
        
    def generate_cross_section_views(self):
        """Generate middle cross-section view with node distribution analysis"""
        
        print("Generating cross-sectional mesh analysis...")
        
        # Clear any existing matplotlib figures to ensure fresh display
        plt.close('all')
        plt.clf()
        
        # Focus only on middle section (heating zone middle)
        middle_z_pos = (config.HEATING_COIL_START + config.HEATING_COIL_END) / 2
        
        # Find Z position where cubic nodes exist
        if self.mesh_data and len(self.mesh_data['z']) > 0:
            # Find closest Z position with cubic nodes
            z_distances = np.abs(self.mesh_data['z'] - middle_z_pos)
            closest_idx = np.argmin(z_distances)
            actual_z_pos = self.mesh_data['z'][closest_idx]
            print(f"   Using Z = {actual_z_pos*1000:.1f}mm (closest to heating zone middle)")
        else:
            actual_z_pos = middle_z_pos
            print(f"   Using Z = {actual_z_pos*1000:.1f}mm (heating zone middle)")
        
        # Create figure with main cross-section and distribution analysis
        fig = plt.figure(figsize=(20, 10))
        
        # Add main title
        fig.suptitle('Tube Furnace Cross-Section Analysis\nHeating Zone Middle', 
                    fontsize=16, fontweight='bold')
        
        # Main cross-section view
        ax1 = fig.add_subplot(1, 3, 1)
        self._plot_cross_section(ax1, actual_z_pos, f'Z = {actual_z_pos*1000:.1f}mm')
        
        # Radial node distribution (single axis)
        ax2 = fig.add_subplot(1, 3, 2)
        self._plot_radial_distribution(ax2, actual_z_pos)
        
        # Node density analysis including radial nodes
        ax3 = fig.add_subplot(1, 3, 3)
        self._plot_complete_node_density(ax3)
        
        plt.suptitle('Tube Furnace Mesh Cross-Sections with Node Distribution', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Force figure update and display
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=True)
        
        return fig
    
    def _plot_cross_section(self, ax, z_position, section_name):
        """Plot cross-section at specific z position with node distribution"""
        
        # Find closest z-index
        z_idx = np.argmin(np.abs(self.mesh.z_nodes - z_position))
        actual_z = self.mesh.z_nodes[z_idx]
        
        # 1. Draw material boundaries (circles for cylindrical region)
        self._draw_cylindrical_boundaries(ax)
        
        # 2. Draw cubic region boundaries
        self._draw_cubic_boundaries(ax)
        
        # 3. Plot cylindrical mesh nodes
        self._plot_cylindrical_nodes(ax, z_idx, actual_z)
        
        # 4. Plot cubic mesh nodes
        self._plot_cubic_nodes(ax, actual_z)
        
        # 5. Add heating coil visualization if in heating zone
        if config.HEATING_COIL_START <= actual_z <= config.HEATING_COIL_END:
            self._draw_heating_coil_cross_section(ax, actual_z)
        
        # Formatting
        max_extent = self.mesh.aluminum_box_outer * 1.1
        ax.set_xlim([-max_extent, max_extent])
        ax.set_ylim([-max_extent, max_extent])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Cross-Section View\n{section_name}', fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        
        # Add layer labels
        self._add_layer_labels(ax)
        
        # Fix legend positioning to avoid clumping with better spacing
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove duplicate entries
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_handles.append(handle)
                    unique_labels.append(label)
            
            # Position legend with proper spacing to prevent clumping
            ax.legend(unique_handles, unique_labels, 
                     bbox_to_anchor=(1.08, 1), loc='upper left', 
                     fontsize=6, frameon=True, fancybox=True,
                     borderaxespad=1.0, columnspacing=1.5, 
                     handletextpad=1.0, handlelength=2.0, labelspacing=1.2)
        
    def _draw_cylindrical_boundaries(self, ax):
        """Draw cylindrical material boundaries"""
        
        boundaries = [
            (self.mesh.sample_radius, self.material_colors['sample_space'], 'Sample Space'),
            (self.mesh.glass_outer_radius, self.material_colors['quartz_glass'], 'Glass Tube'),
            (self.mesh.cement_outer_radius, self.material_colors['imperial_cement'], 'Cement'),
            (self.mesh.ceramic_outer_radius, self.material_colors['lyrufexon_ceramic'], 'Ceramic Wool'),
            (self.mesh.reflective_outer_radius, self.material_colors['reflective_aluminum'], 'Reflective Al')
        ]
        
        for radius, color, label in boundaries:
            if radius > 0:
                circle = Circle((0, 0), radius, fill=False, edgecolor=color, 
                              linewidth=2, linestyle='-', label=label)
                ax.add_patch(circle)
    
    def _draw_cubic_boundaries(self, ax):
        """Draw cubic region boundaries"""
        
        # Air gap boundary (square)
        air_gap_size = self.mesh.air_gap_outer
        air_gap_rect = Rectangle((-air_gap_size, -air_gap_size), 
                                2*air_gap_size, 2*air_gap_size,
                                fill=False, edgecolor=self.material_colors['air_gap'],
                                linewidth=2, linestyle='--', label='Air Gap Boundary')
        ax.add_patch(air_gap_rect)
        
        # Aluminum box boundary (outer square)
        box_size = self.mesh.aluminum_box_outer
        box_rect = Rectangle((-box_size, -box_size), 
                            2*box_size, 2*box_size,
                            fill=False, edgecolor=self.material_colors['aluminum_5052'],
                            linewidth=3, linestyle='-', label='Aluminum Box')
        ax.add_patch(box_rect)
    
    def _plot_cylindrical_nodes(self, ax, z_idx, z_position):
        """Plot cylindrical mesh nodes"""
        
        # Sample every few nodes for clarity
        r_sample_step = max(1, len(self.mesh.r_nodes) // 20)  # Show ~20 radial nodes
        
        for i in range(0, len(self.mesh.r_nodes), r_sample_step):
            r = self.mesh.r_nodes[i]
            
            # Determine material and color
            if r <= self.mesh.sample_radius:
                color = 'lightblue'
                size = 8
            elif r <= self.mesh.glass_outer_radius:
                color = 'gray'
                size = 8
            elif r <= self.mesh.cement_outer_radius:
                color = 'orange'
                size = 10
            elif r <= self.mesh.ceramic_outer_radius:
                color = 'gold'
                size = 8
            else:
                color = 'silver'
                size = 8
            
            # Plot nodes in single direction only (along X-axis)
            # Show nodes at ±r along X-axis and at center if r=0
            if r == 0:
                ax.scatter(0, 0, c=color, s=size, alpha=0.8, 
                          edgecolors='black', linewidth=0.5, zorder=5)
            else:
                # Plot along X-axis only (two points: +r and -r)
                ax.scatter([r, -r], [0, 0], c=color, s=size, alpha=0.8, 
                          edgecolors='black', linewidth=0.5, zorder=5)
    
    def _plot_cubic_nodes(self, ax, z_position):
        """Plot cubic mesh nodes at this z position using current mesh data"""
        
        if not self.mesh_data or len(self.mesh_data['x']) == 0:
            print("   No cubic mesh data available for plotting")
            return
            
        # Find cubic nodes at this z position (within tolerance)
        z_tolerance = 0.01  # 10mm tolerance for z-axis (in meters)
        z_indices = np.where(np.abs(self.mesh_data['z'] - z_position) <= z_tolerance)[0]
        
        print(f"   Looking for cubic nodes at Z={z_position*1000:.1f}mm with ±{z_tolerance*1000:.1f}mm tolerance")
        print(f"   Found {len(z_indices)} cubic nodes at this Z position")
        
        if len(z_indices) == 0:
            # Try to find the closest Z position
            z_distances = np.abs(self.mesh_data['z'] - z_position)
            closest_idx = np.argmin(z_distances)
            closest_z = self.mesh_data['z'][closest_idx]
            print(f"   No nodes found. Closest Z position is {closest_z*1000:.1f}mm")
            return
            
        x_slice = self.mesh_data['x'][z_indices]
        y_slice = self.mesh_data['y'][z_indices]
        materials = self.mesh_data['material'][z_indices]
        
        # Separate by material type
        air_gap_mask = materials == 'air_gap'
        aluminum_mask = materials == 'aluminum_5052'
        
        print(f"   Air gap nodes: {np.sum(air_gap_mask)}")
        print(f"   Aluminum nodes: {np.sum(aluminum_mask)}")
        
        # Plot air gap nodes (light green)
        if np.any(air_gap_mask):
            ax.scatter(x_slice[air_gap_mask], y_slice[air_gap_mask], 
                      c='lightgreen', s=12, alpha=0.8, label='Air Gap Nodes',
                      edgecolors='darkgreen', linewidth=0.5, zorder=3)
        
        # Plot aluminum nodes (gray)
        if np.any(aluminum_mask):
            ax.scatter(x_slice[aluminum_mask], y_slice[aluminum_mask], 
                      c='gray', s=15, alpha=0.9, label='Aluminum Nodes',
                      edgecolors='black', linewidth=0.5, zorder=4)
    
    def _draw_heating_coil_cross_section(self, ax, z_position):
        """Draw heating coil cross-section representation"""
        
        coil_radius = config.HEATING_COIL_RADIUS
        wire_diameter = config.HEATING_COIL_WIRE_DIAMETER
        
        # Calculate coil turn angle at this z position
        turns_per_length = config.HEATING_COIL_TURNS / config.HEATING_COIL_LENGTH
        z_offset = z_position - config.HEATING_COIL_START
        turn_angle = 2 * np.pi * turns_per_length * z_offset
        
        # Position of wire center at this z
        wire_x = coil_radius * np.cos(turn_angle)
        wire_y = coil_radius * np.sin(turn_angle)
        
        # Draw wire cross-section
        wire_circle = Circle((wire_x, wire_y), wire_diameter/2, 
                           fill=True, facecolor=self.material_colors['heating_coil'],
                           edgecolor='darkred', linewidth=2, alpha=0.9, zorder=10)
        ax.add_patch(wire_circle)
        
        # Add wire label
        ax.annotate('Heating Wire', xy=(wire_x, wire_y), 
                   xytext=(wire_x + 0.02, wire_y + 0.02),
                   fontsize=8, color='red', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    def _add_cross_section_info(self, ax):
        """Add information panel for cross-sections"""
        
        ax.axis('off')
        
        # Title
        ax.text(0.02, 0.95, 'CROSS-SECTION ANALYSIS', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        
        y_pos = 0.85
        line_spacing = 0.06
        
        # Node counts
        cylindrical_nodes_per_section = len(self.mesh.r_nodes)
        
        ax.text(0.02, y_pos, 'NODE DISTRIBUTION PER SECTION:', fontsize=12, fontweight='bold',
                transform=ax.transAxes, color='darkblue')
        y_pos -= line_spacing * 0.8
        
        ax.text(0.05, y_pos, f'• Cylindrical nodes: {cylindrical_nodes_per_section}',
                fontsize=10, transform=ax.transAxes)
        y_pos -= line_spacing * 0.7
        
        # Estimate cubic nodes per section
        if hasattr(self.mesh, 'cubic_nodes_xyz'):
            cubic_nodes_per_section = len(self.mesh.cubic_nodes_xyz) // len(self.mesh.z_nodes)
            ax.text(0.05, y_pos, f'• Cubic nodes: ~{cubic_nodes_per_section}',
                    fontsize=10, transform=ax.transAxes)
        y_pos -= line_spacing
        
        # Material regions
        ax.text(0.02, y_pos, 'MATERIAL REGIONS:', fontsize=12, fontweight='bold',
                transform=ax.transAxes, color='darkgreen')
        y_pos -= line_spacing * 0.8
        
        materials = [
            ('Sample Space', self.material_colors['sample_space']),
            ('Glass Tube', self.material_colors['quartz_glass']),
            ('Cement + Coil', self.material_colors['imperial_cement']),
            ('Ceramic Wool', self.material_colors['lyrufexon_ceramic']),
            ('Reflective Al', self.material_colors['reflective_aluminum']),
            ('Air Gap', self.material_colors['air_gap']),
            ('Aluminum Box', self.material_colors['aluminum_5052'])
        ]
        
        for material, color in materials:
            ax.text(0.05, y_pos, f'• {material}', fontsize=9,
                   transform=ax.transAxes, color='white' if color in ['#8B4513', '#696969'] else 'black',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8, edgecolor='black'))
            y_pos -= line_spacing * 0.6
        
        y_pos -= line_spacing * 0.5
        
        # Physical dimensions
        ax.text(0.02, y_pos, 'PHYSICAL DIMENSIONS:', fontsize=12, fontweight='bold',
                transform=ax.transAxes, color='purple')
        y_pos -= line_spacing * 0.8
        
        dimensions = [
            f'Sample radius: {self.mesh.sample_radius*1000:.1f}mm',
            f'Glass outer: {self.mesh.glass_outer_radius*1000:.1f}mm',
            f'Cement outer: {self.mesh.cement_outer_radius*1000:.1f}mm',
            f'Ceramic outer: {self.mesh.ceramic_outer_radius*1000:.1f}mm',
            f'Reflective outer: {self.mesh.reflective_outer_radius*1000:.1f}mm',
            f'Air gap outer: {self.mesh.air_gap_outer*1000:.1f}mm',
            f'Box outer: {self.mesh.aluminum_box_outer*1000:.1f}mm'
        ]
        
        for dim in dimensions:
            ax.text(0.05, y_pos, f'• {dim}', fontsize=9,
                   transform=ax.transAxes)
            y_pos -= line_spacing * 0.6

    def _plot_radial_distribution(self, ax, z_pos):
        """Plot radial node distribution along single X-axis from center outward (positive direction only)"""
        # Get cubic nodes at this z position (within tolerance)
        z_tolerance = 0.01  # 10mm tolerance (in meters)
        z_indices = np.where(np.abs(self.mesh_data['z'] - z_pos) <= z_tolerance)[0]
        
        if len(z_indices) == 0:
            ax.text(0.5, 0.5, 'No cubic nodes at this Z position', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Radial Node Distribution')
            return
            
        x_slice = self.mesh_data['x'][z_indices]
        y_slice = self.mesh_data['y'][z_indices]
        materials = self.mesh_data['material'][z_indices]
        
        # Find nodes along X-axis only (y ≈ 0) - POSITIVE DIRECTION ONLY
        tolerance = 0.005  # 5mm tolerance for Y coordinate (in meters)
        x_axis_mask = np.abs(y_slice) <= tolerance
        positive_x_mask = x_slice >= 0  # Only positive X direction
        combined_mask = x_axis_mask & positive_x_mask
        
        if not np.any(combined_mask):
            ax.text(0.5, 0.5, 'No nodes along positive X-axis at this Z position', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Radial Node Distribution')
            return
        
        # Get X positions and materials for nodes along positive X-axis
        x_positions = x_slice[combined_mask] * 1000  # Convert to mm
        x_materials = materials[combined_mask]
        
        # Separate by material type
        air_gap_mask = x_materials == 'air_gap'
        aluminum_mask = x_materials == 'aluminum_5052'
        
        # Plot cubic nodes based on their actual spatial XY distribution
        # Show nodes spreading right (X) and up (Y) based on coordinates
        if len(z_indices) > 0:
            # Get all cubic nodes at this Z level (not just X-axis)
            x_all = x_slice * 1000  # Convert to mm
            y_all = y_slice * 1000  # Convert to mm
            materials_all = materials
            
            # Filter for positive X and positive Y quadrant only
            positive_quadrant_mask = (x_all >= 0) & (y_all >= 0)
            
            if np.any(positive_quadrant_mask):
                x_quad = x_all[positive_quadrant_mask]
                y_quad = y_all[positive_quadrant_mask] 
                materials_quad = materials_all[positive_quadrant_mask]
                
                # Separate by material type and plot with spatial distribution
                air_gap_mask = materials_quad == 'air_gap'
                aluminum_mask = materials_quad == 'aluminum_5052'
                
                # Calculate cubic region Y positions using the same scale factor
                scale_factor = 2.0 / (self.mesh.reflective_outer_radius * 1000)
                reflective_y = self.mesh.reflective_outer_radius * 1000 * scale_factor
                air_gap_y = reflective_y + 0.4
                aluminum_y = air_gap_y + 0.3
                
                if np.any(air_gap_mask):
                    # Map Y coordinates to air gap level with spatial distribution
                    max_y = np.max(y_quad[air_gap_mask])
                    min_y = np.min(y_quad[air_gap_mask])
                    y_range = max_y - min_y if max_y > min_y else 1.0
                    
                    # Scale Y coordinates to spread between reflective and air gap legend
                    y_normalized = (y_quad[air_gap_mask] - min_y) / y_range
                    y_display = reflective_y + y_normalized * (air_gap_y - reflective_y)
                    
                    ax.scatter(x_quad[air_gap_mask], y_display, 
                              c='lightgreen', s=60, alpha=0.8, label='Air Gap Nodes',
                              edgecolors='darkgreen', linewidth=1)
                
                if np.any(aluminum_mask):
                    # Map Y coordinates to aluminum level with spatial distribution
                    max_y = np.max(y_quad[aluminum_mask])
                    min_y = np.min(y_quad[aluminum_mask])
                    y_range = max_y - min_y if max_y > min_y else 1.0
                    
                    # Scale Y coordinates to spread around aluminum legend level
                    y_normalized = (y_quad[aluminum_mask] - min_y) / y_range
                    y_display = air_gap_y + y_normalized * (aluminum_y - air_gap_y)
                    
                    ax.scatter(x_quad[aluminum_mask], y_display, 
                              c='gray', s=60, alpha=0.9, label='Aluminum Nodes',
                              edgecolors='black', linewidth=1)
        
        # Add cylindrical region nodes with proper radial density distribution
        # Show higher density near heating coil (cement region where wire is embedded)
        self._plot_cylindrical_radial_density(ax)
        
        # Add region boundaries as vertical lines (positive direction only)
        ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='CENTER (Y=0)')
        ax.axhline(0, color='black', linestyle='-', linewidth=3, alpha=0.8, label='SAMPLE STARTS HERE')
        ax.axvline(self.mesh.cylindrical_outer_radius * 1000, color='red', 
                  linestyle=':', alpha=0.7, label='Cylindrical Boundary')
        ax.axvline(self.mesh.air_gap_outer * 1000, color='orange',
                  linestyle='--', alpha=0.7, label='Air Gap Boundary')
        ax.axvline(self.mesh.aluminum_box_outer * 1000, color='purple',
                  linestyle='-', alpha=0.8, label='Aluminum Box Outer')
        
        # Formatting
        ax.set_xlabel('Radial Position (mm)')
        ax.set_ylabel('Radial Distance from Center (scaled)')
        ax.set_title('Radial Node Distribution\n(Nodes positioned by radial distance - Positive Direction)')
        ax.set_ylim(-0.1, 3.2)  # Extended range to provide more space for legends
        ax.set_xlim(0, 160)  # Only positive direction
        
        # Custom y-ticks calculated from actual physical dimensions in config
        # Scale factor to convert mm to Y-axis units (adjust for visual clarity)
        scale_factor = 2.0 / (self.mesh.reflective_outer_radius * 1000)  # Normalize to Y=2.0 for reflective boundary
        
        # Calculate Y positions based on actual radial boundaries - Sample starts at center (0)
        center_y = 0.0  # Sample starts at center
        sample_y = self.mesh.sample_radius * 1000 * scale_factor
        glass_y = self.mesh.glass_outer_radius * 1000 * scale_factor  
        cement_y = self.mesh.cement_outer_radius * 1000 * scale_factor
        ceramic_y = self.mesh.ceramic_outer_radius * 1000 * scale_factor
        reflective_y = self.mesh.reflective_outer_radius * 1000 * scale_factor
        air_gap_y = reflective_y + 0.4  # Offset for cubic region
        aluminum_y = air_gap_y + 0.3   # Final level
        
        print(f"   DEBUG: Y-axis positions calculated:")
        print(f"     Center (Y=0): {center_y:.2f}")
        print(f"     Sample: {sample_y:.2f}")
        print(f"     Reflective: {reflective_y:.2f}")
        print(f"     Air Gap: {air_gap_y:.2f}")
        print(f"     Aluminum: {aluminum_y:.2f}")
        
        y_positions = [center_y, sample_y, glass_y, cement_y, ceramic_y, reflective_y, air_gap_y, aluminum_y]
        y_labels = ['Center', 'Sample', 'Glass', 'Cement\n(Heating)', 'Ceramic', 'Reflective', 'Air Gap', 'Aluminum']
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=8)
        
        # Fix legend positioning with better spacing to prevent clumping
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Remove duplicates and improve spacing
            unique_handles = []
            unique_labels = []
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_handles.append(handle)
                    unique_labels.append(label)
            
            ax.legend(unique_handles, unique_labels, fontsize=6, 
                     bbox_to_anchor=(1.08, 1), loc='upper left',
                     frameon=True, fancybox=True, borderaxespad=1.0,
                     columnspacing=1.5, handletextpad=1.0, handlelength=2.0,
                     labelspacing=1.2)
        ax.grid(True, alpha=0.3)

    def _plot_region_based_analysis(self, ax):
        """Plot node density across physical regions with actual dimensions"""
        
        # Physical region boundaries (in mm)
        regions = [
            {
                'name': 'Near\nReflective',
                'start': self.mesh.cylindrical_outer_radius * 1000,  # 118.5mm
                'end': 123.0,  # Approximate boundary
                'nodes': config.AIR_GAP_NEAR_REFLECTIVE_NODES_PER_SIDE,
                'color': 'lightblue',
                'material': 'Air Gap'
            },
            {
                'name': 'Middle\nAir Gap', 
                'start': 123.0,
                'end': 139.0,
                'nodes': config.AIR_GAP_MIDDLE_NODES_PER_SIDE,
                'color': 'lightgreen',
                'material': 'Air Gap'
            },
            {
                'name': 'Near\nAluminum',
                'start': 139.0,
                'end': self.mesh.air_gap_outer * 1000,  # 148.5mm
                'nodes': config.AIR_GAP_NEAR_ALUMINUM_NODES_PER_SIDE,
                'color': 'orange', 
                'material': 'Air Gap'
            },
            {
                'name': 'Aluminum\nSurface',
                'start': self.mesh.air_gap_outer * 1000,  # 148.5mm
                'end': self.mesh.aluminum_box_outer * 1000,  # 151.5mm
                'nodes': config.ALUMINUM_BOX_SURFACE_NODES_PER_SIDE,
                'color': 'gray',
                'material': 'Aluminum'
            }
        ]
        
        # Calculate node density (nodes per mm) for each region
        positions = []
        densities = []
        colors = []
        widths = []
        labels = []
        
        for region in regions:
            width = region['end'] - region['start']  # mm
            center_pos = (region['start'] + region['end']) / 2
            
            # Node density = nodes per mm width
            if width > 0:
                density = region['nodes'] / width
            else:
                density = 0
                
            positions.append(center_pos)
            densities.append(density)
            colors.append(region['color'])
            widths.append(width * 0.8)  # Make bars slightly narrower than region
            labels.append(region['name'])
        
        # Create bar chart showing node density
        bars = ax.bar(positions, densities, width=widths, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, density, region) in enumerate(zip(bars, densities, regions)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{density:.3f}\nnodes/mm', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold')
            
            # Add node count at bottom
            ax.text(bar.get_x() + bar.get_width()/2., 0.001,
                   f'{region["nodes"]} nodes', ha='center', va='bottom',
                   fontsize=7, color='darkblue')
        
        # Customize the plot
        ax.set_xlabel('Radial Position (mm)')
        ax.set_ylabel('Node Density (nodes/mm)')
        ax.set_title('Node Density Distribution\nAcross Physical Regions')
        
        # Set x-axis limits and ticks
        ax.set_xlim(115, 155)
        ax.set_xticks([120, 130, 140, 150])
        
        # Add region boundary lines
        for region in regions:
            ax.axvline(region['start'], color='red', linestyle=':', alpha=0.5)
        ax.axvline(regions[-1]['end'], color='red', linestyle=':', alpha=0.5)
        
        # Add grid
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend for materials
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='Air Gap'),
            Patch(facecolor='gray', alpha=0.7, label='Aluminum')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    def _plot_cylindrical_radial_density(self, ax):
        """Plot cylindrical nodes with realistic radial density distribution"""
        
        # Define layer positions using actual physical dimensions from config
        # Scale factor to match Y-axis scaling in main plot
        scale_factor = 2.0 / (self.mesh.reflective_outer_radius * 1000)
        
        layers = [
            {'name': 'Sample', 'radius': self.mesh.sample_radius * 1000, 
             'y_pos': self.mesh.sample_radius * 1000 * scale_factor, 'nodes': config.RADIAL_NODES_SAMPLE, 'color': 'lightblue'},
            {'name': 'Glass', 'radius': self.mesh.glass_outer_radius * 1000, 
             'y_pos': self.mesh.glass_outer_radius * 1000 * scale_factor, 'nodes': config.RADIAL_NODES_GLASS, 'color': 'lightgray'},
            {'name': 'Cement + Heating Wire', 'radius': self.mesh.cement_outer_radius * 1000, 
             'y_pos': self.mesh.cement_outer_radius * 1000 * scale_factor, 'nodes': config.RADIAL_NODES_CEMENT, 'color': 'orange'},
            {'name': 'Ceramic', 'radius': self.mesh.ceramic_outer_radius * 1000, 
             'y_pos': self.mesh.ceramic_outer_radius * 1000 * scale_factor, 'nodes': config.RADIAL_NODES_CERAMIC, 'color': 'gold'},
            {'name': 'Reflective Al', 'radius': self.mesh.reflective_outer_radius * 1000, 
             'y_pos': self.mesh.reflective_outer_radius * 1000 * scale_factor, 'nodes': config.RADIAL_NODES_REFLECTIVE, 'color': 'silver'}
        ]
        
        # Plot nodes for each cylindrical layer accurately distributed across radial dimensions
        for layer in layers:
            num_nodes = layer['nodes']
            layer_idx = layers.index(layer)
            
            # Get accurate radial boundaries for this layer
            if layer_idx == 0:  # Sample layer
                r_inner = 0
                r_outer = layer['radius']
            else:
                r_inner = layers[layer_idx - 1]['radius']
                r_outer = layer['radius']
            
            # Distribute nodes accurately across the radial thickness of each layer
            if num_nodes == 1:
                # Single node at mid-radius
                r_mid = (r_inner + r_outer) / 2
                x_positions = [r_mid]
                # Y position corresponds to the layer's legend position
                y_positions = [layer['y_pos']]
            else:
                # Multiple nodes distributed across radial thickness
                if layer['name'] == 'Sample':
                    # Sample: from center to outer boundary
                    r_positions = np.linspace(r_inner, r_outer, num_nodes)
                    # Distribute Y positions from 0 (center) to sample legend position
                    y_positions = np.linspace(0, layer['y_pos'], num_nodes).tolist()
                else:
                    # Other layers: evenly distributed across layer thickness
                    r_positions = np.linspace(r_inner, r_outer, num_nodes)
                    # Get previous layer Y position for proper scaling
                    prev_y = layers[layer_idx - 1]['y_pos'] if layer_idx > 0 else 0
                    # Distribute Y positions from previous layer to current layer
                    y_positions = np.linspace(prev_y, layer['y_pos'], num_nodes).tolist()
                
                x_positions = r_positions.tolist()
            
            # Special highlighting for heating element region (cement layer)
            if 'Heating' in layer['name']:
                # Higher density and special marker for heating region
                ax.scatter(x_positions, y_positions, 
                          c=layer['color'], s=120, alpha=0.9, marker='^',
                          edgecolors='red', linewidth=2, 
                          label=f"{layer['name']} ({layer['nodes']} nodes)")
                
                # Add heating zone indicator - properly sized for the cement layer (Y=0.2 to Y=0.4)
                prev_y = layers[layer_idx - 1]['y_pos'] if layer_idx > 0 else 0
                ax.axhspan(prev_y, layer['y_pos'], 
                          color='red', alpha=0.15, label='Heating Element Zone')
            else:
                ax.scatter(x_positions, y_positions, 
                          c=layer['color'], s=60, alpha=0.8,
                          edgecolors='black', linewidth=1, 
                          label=f"{layer['name']} ({layer['nodes']} nodes)")

    def _add_layer_labels(self, ax):
        """Add labels for different material layers"""
        
        # Text positions (on right side of plot)
        label_x = self.mesh.aluminum_box_outer * 0.7
        
        # Cylindrical layer labels
        layers = [
            (self.mesh.sample_radius * 0.5, 'Sample'),
            ((self.mesh.sample_radius + self.mesh.glass_outer_radius) * 0.5, 'Glass'),
            ((self.mesh.glass_outer_radius + self.mesh.cement_outer_radius) * 0.5, 'Cement'),
            ((self.mesh.cement_outer_radius + self.mesh.ceramic_outer_radius) * 0.5, 'Ceramic'),
            ((self.mesh.ceramic_outer_radius + self.mesh.reflective_outer_radius) * 0.5, 'Reflective Al'),
            ((self.mesh.cylindrical_outer_radius + self.mesh.air_gap_outer) * 0.5, 'Air Gap'),
            ((self.mesh.air_gap_outer + self.mesh.aluminum_box_outer) * 0.5, 'Al Box')
        ]
        
        for radius, label in layers:
            if radius < self.mesh.aluminum_box_outer:
                ax.text(label_x, radius, label, fontsize=8, ha='left', va='center',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

    def _plot_complete_node_density(self, ax):
        """Plot complete node density including both cylindrical and cubic regions"""
        
        # Define all physical regions from center outward
        regions = [
            # Cylindrical regions (from config)
            {
                'name': 'Sample',
                'start': 0.0,
                'end': self.mesh.sample_radius * 1000,
                'nodes': config.RADIAL_NODES_SAMPLE,
                'color': 'lightblue',
                'region_type': 'Cylindrical'
            },
            {
                'name': 'Glass',
                'start': self.mesh.sample_radius * 1000,
                'end': self.mesh.glass_outer_radius * 1000,
                'nodes': config.RADIAL_NODES_GLASS,
                'color': 'lightgray',
                'region_type': 'Cylindrical'
            },
            {
                'name': 'Cement',
                'start': self.mesh.glass_outer_radius * 1000,
                'end': self.mesh.cement_outer_radius * 1000,
                'nodes': config.RADIAL_NODES_CEMENT,
                'color': 'orange',
                'region_type': 'Cylindrical'
            },
            {
                'name': 'Ceramic',
                'start': self.mesh.cement_outer_radius * 1000,
                'end': self.mesh.ceramic_outer_radius * 1000,
                'nodes': config.RADIAL_NODES_CERAMIC,
                'color': 'gold',
                'region_type': 'Cylindrical'
            },
            {
                'name': 'Reflective',
                'start': self.mesh.ceramic_outer_radius * 1000,
                'end': self.mesh.reflective_outer_radius * 1000,
                'nodes': config.RADIAL_NODES_REFLECTIVE,
                'color': 'silver',
                'region_type': 'Cylindrical'
            },
            # Cubic regions (from config)
            {
                'name': 'Near\nReflective',
                'start': self.mesh.cylindrical_outer_radius * 1000,
                'end': 123.0,
                'nodes': config.AIR_GAP_NEAR_REFLECTIVE_NODES_PER_SIDE,
                'color': 'lightcyan',
                'region_type': 'Cubic'
            },
            {
                'name': 'Middle\nAir Gap', 
                'start': 123.0,
                'end': 139.0,
                'nodes': config.AIR_GAP_MIDDLE_NODES_PER_SIDE,
                'color': 'lightgreen',
                'region_type': 'Cubic'
            },
            {
                'name': 'Near\nAluminum',
                'start': 139.0,
                'end': self.mesh.air_gap_outer * 1000,
                'nodes': config.AIR_GAP_NEAR_ALUMINUM_NODES_PER_SIDE,
                'color': 'lightyellow',
                'region_type': 'Cubic'
            },
            {
                'name': 'Aluminum\nBox',
                'start': self.mesh.air_gap_outer * 1000,
                'end': self.mesh.aluminum_box_outer * 1000,
                'nodes': config.ALUMINUM_BOX_SURFACE_NODES_PER_SIDE,
                'color': 'gray',
                'region_type': 'Cubic'
            }
        ]
        
        # Calculate node density for each region
        positions = []
        densities = []
        colors = []
        widths = []
        labels = []
        
        for region in regions:
            width = region['end'] - region['start']  # mm
            center_pos = (region['start'] + region['end']) / 2
            
            # Node density = nodes per mm width
            if width > 0:
                density = region['nodes'] / width
            else:
                density = 0
                
            positions.append(center_pos)
            densities.append(density)
            colors.append(region['color'])
            widths.append(width * 0.8)  # Make bars slightly narrower
            labels.append(region['name'])
        
        # Create bar chart
        bars = ax.bar(positions, densities, width=widths, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Create bar chart
        bars = ax.bar(positions, densities, width=widths, color=colors, 
                     alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for i, (bar, density, region) in enumerate(zip(bars, densities, regions)):
            height = bar.get_height()
            # Only show density for non-zero heights
            if height > 0.001:
                # Top number: node density (nodes/mm)
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{density:.3f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold', color='red')
            
            # Bottom number: total node count in region
            ax.text(bar.get_x() + bar.get_width()/2., 0.005,
                   f'{region["nodes"]}', ha='center', va='bottom',
                   fontsize=8, color='darkblue', fontweight='bold')
            
            # Add region name labels below x-axis
            ax.text(bar.get_x() + bar.get_width()/2., -0.05,
                   region['name'], ha='center', va='top',
                   fontsize=7, rotation=45)
        
        # Add region boundary markers
        ax.axvline(self.mesh.cylindrical_outer_radius * 1000, color='red', 
                  linestyle='--', linewidth=2, alpha=0.8, 
                  label='Cylindrical/Cubic Boundary')
        
        # Customize plot
        ax.set_xlabel('Radial Position from Center (mm)')
        ax.set_ylabel('Node Density (nodes/mm)')
        ax.set_title('Complete Node Density Distribution\n(Cylindrical + Cubic Regions)')
        ax.set_xlim(-5, 160)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add legend with proper positioning to avoid clumping
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Create legend elements in organized groups
        region_legend = [
            Patch(facecolor='lightblue', alpha=0.8, label='Cylindrical Regions'),
            Patch(facecolor='lightgreen', alpha=0.8, label='Cubic Regions')
        ]
        
        number_legend = [
            Line2D([0], [0], color='red', marker='o', markersize=5, 
                   label='Red: Density (nodes/mm)', linestyle='None'),
            Line2D([0], [0], color='darkblue', marker='s', markersize=5,
                   label='Blue: Node Count', linestyle='None')
        ]
        
        boundary_legend = [ax.lines[-1]]  # The boundary line
        
        # Combine all legend elements with proper spacing
        all_legend_elements = region_legend + number_legend + boundary_legend
        
        # Position legend outside plot area with proper spacing to prevent clumping
        ax.legend(handles=all_legend_elements, 
                 bbox_to_anchor=(1.08, 1), loc='upper left',
                 fontsize=6, frameon=True, fancybox=True, 
                 shadow=False, ncol=1, borderaxespad=1.0,
                 columnspacing=1.5, handletextpad=1.0, labelspacing=1.2)

def generate_detailed_cross_sections():
    """Main function to generate detailed cross-section analysis"""
    
    print("Starting detailed cross-section analysis...")
    
    analyzer = MeshCrossSectionAnalyzer()
    fig = analyzer.generate_cross_section_views()
    
    print("Cross-section analysis complete!")
    return analyzer, fig

if __name__ == "__main__":
    analyzer, fig = generate_detailed_cross_sections()
    print("Cross-section visualization complete. Close the plot to continue.")