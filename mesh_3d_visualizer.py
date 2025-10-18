"""
3D Mesh Visualizer for Tube Furnace Mesh
Shows cylindrical and cubic regions in 3D space
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import config
from mesh import TubeFurnaceMesh

def visualize_3d_mesh():
    """Create comprehensive 3D visualization of the correct tube furnace mesh"""
    
    print("Creating 3D visualization of correct tube furnace mesh...")
    
    # Generate the mesh
    mesh = TubeFurnaceMesh()
    mesh.generate_complete_mesh()
    
    # Create 3D plot with better layout
    fig = plt.figure(figsize=(20, 12))
    
    # Create subplot layout: 3D plot on left, legend/info on right
    ax = fig.add_subplot(121, projection='3d')  # Left side for 3D plot
    info_ax = fig.add_subplot(122)  # Right side for legend and information
    info_ax.axis('off')  # Turn off axes for info panel
    
    print("\nVisualizing mesh components...")
    
    # ==================== CYLINDRICAL REGION VISUALIZATION ====================
    print("1. Plotting cylindrical region boundaries and radial nodes...")
    
    # Create full cylindrical surfaces for 3D visualization (full 360°)
    theta = np.linspace(0, 2*np.pi, 32)  # Full circle, 32 points
    z_cyl = np.linspace(0, config.FURNACE_LENGTH, 20)  # 20 points along length
    
    THETA, Z_CYL = np.meshgrid(theta, z_cyl)
    
    # Plot key cylindrical boundaries (sample space removed per user request)
    boundaries = [
        (mesh.glass_outer_radius, 'Glass tube', 'cyan', 0.4), 
        (mesh.cement_outer_radius, 'Cement', 'orange', 0.3),
        (mesh.ceramic_outer_radius, 'Ceramic wool', 'yellow', 0.2),
        (mesh.reflective_outer_radius, 'Reflective Al', 'silver', 0.6)
    ]
    
    for radius, label, color, alpha in boundaries:
        if radius > 0:  # Skip centerline
            X_cyl = radius * np.cos(THETA)
            Y_cyl = radius * np.sin(THETA)
            ax.plot_surface(X_cyl, Y_cyl, Z_CYL, alpha=alpha, color=color, label=label)
    
    # Add radial mesh nodes visualization - SINGLE RADIAL DIRECTION ONLY
    print("   Adding radial mesh nodes (single direction)...")
    if hasattr(mesh, 'r_nodes') and mesh.r_nodes is not None:
        # Sample radial and axial nodes for visualization
        r_sample = mesh.r_nodes[::2]  # Every 2nd radial node for better resolution
        z_sample = mesh.z_nodes[::3]  # Every 3rd axial node
        
        # Show nodes along SINGLE radial direction only (positive X-axis, θ=0)
        theta_single = 0.0  # Only along positive X-axis
        
        radial_node_count = 0
        for r in r_sample:
            if r > 0:  # Skip center point
                for z in z_sample:
                    x = r * np.cos(theta_single)  # Always along X-axis
                    y = r * np.sin(theta_single)  # Y = 0
                    
                    # Color nodes by material region (yellow for ceramic as mentioned)
                    if r <= mesh.glass_outer_radius:
                        color = 'lightblue'
                        size = 6
                    elif r <= mesh.cement_outer_radius:
                        color = 'orange' 
                        size = 6
                    elif r <= mesh.ceramic_outer_radius:
                        color = 'yellow'  # Yellow dots as requested
                        size = 8
                    else:
                        color = 'silver'
                        size = 6
                    
                    ax.scatter(x, y, z, c=color, s=size, alpha=0.9, edgecolors='black', linewidth=0.5)
                    radial_node_count += 1
        
        print(f"   Plotted {radial_node_count} radial nodes along single direction (from {len(mesh.r_nodes) * len(mesh.z_nodes):,} total)")
    else:
        print("   No radial nodes available for plotting")
    
    # ==================== CUBIC REGION VISUALIZATION ====================
    print("2. Plotting cubic region nodes (axisymmetric view)...")
    
    # Plot cubic mesh nodes using full 3D array representation
    if hasattr(mesh, 'cubic_nodes_xyz') and mesh.cubic_nodes_xyz is not None:
        print(f"   Plotting full 3D cubic array...")
        
        # Sample nodes for better performance but show full 3D structure
        step = 12  # Sample every 12th node for performance
        cubic_sample = mesh.cubic_nodes_xyz[::step]
        
        # Separate nodes by region - show full 3D array
        air_gap_nodes = []
        aluminum_nodes = []
        
        for x, y, z in cubic_sample:
            # Check if in aluminum wall region
            if abs(x) > mesh.air_gap_outer or abs(y) > mesh.air_gap_outer:
                aluminum_nodes.append([x, y, z])
            else:
                air_gap_nodes.append([x, y, z])
        
        # Plot air gap nodes (green)
        if air_gap_nodes:
            air_gap_array = np.array(air_gap_nodes)
            ax.scatter(air_gap_array[:, 0], air_gap_array[:, 1], air_gap_array[:, 2], 
                      c='lightgreen', s=6, alpha=0.7, label='Air gap nodes')
        
        # Plot aluminum nodes (gray)
        if aluminum_nodes:
            aluminum_array = np.array(aluminum_nodes)
            ax.scatter(aluminum_array[:, 0], aluminum_array[:, 1], aluminum_array[:, 2], 
                      c='gray', s=6, alpha=0.8, label='Aluminum nodes')
        
        print(f"   Plotted {len(air_gap_nodes)} air gap + {len(aluminum_nodes)} aluminum nodes (full 3D)")
    else:
        print("   No cubic nodes available for plotting")
        
    # ==================== HEATING ZONE VISUALIZATION ====================
    print("3. Highlighting heating zone with turn information...")
    
    # Mark heating zone boundaries
    heating_start_z = config.HEATING_COIL_START
    heating_end_z = config.HEATING_COIL_END
    
    # Create heating zone cylinder
    r_heating = mesh.cement_outer_radius * 0.8  # Slightly inside cement
    X_heat = r_heating * np.cos(THETA)
    Y_heat = r_heating * np.sin(THETA)
    Z_heat_start = np.full_like(THETA, heating_start_z)
    Z_heat_end = np.full_like(THETA, heating_end_z)
    
    # Plot heating zone boundaries
    for i in range(len(theta)):
        ax.plot([X_heat[0,i], X_heat[0,i]], [Y_heat[0,i], Y_heat[0,i]], 
               [heating_start_z, heating_end_z], 'r-', linewidth=2, alpha=0.8)
    
    # Add heating coil visualization with turn markers (axisymmetric view)
    total_turns = config.HEATING_COIL_TURNS
    coil_radius = config.HEATING_COIL_RADIUS
    
    # Create helix for heating coil visualization - full 3D helix
    helix_turns = np.linspace(0, total_turns * 2 * np.pi, 300)  # More points for smoother full helix
    helix_z = np.linspace(heating_start_z, heating_end_z, 300)
    helix_x = coil_radius * np.cos(helix_turns)
    helix_y = coil_radius * np.sin(helix_turns)
    
    # Plot full heating coil as helix (complete 3D)
    ax.plot(helix_x, helix_y, helix_z, 'red', linewidth=3, alpha=0.9, 
           label=f'Heating Coil ({total_turns} turns)')
    
    # Mark turn positions along the coil (full 3D)
    turn_positions = np.linspace(0, total_turns, int(total_turns) + 1)
    for i, turn in enumerate(turn_positions):
        if i % 10 == 0:  # Mark every 10th turn to avoid clutter
            angle = turn * 2 * np.pi
            z_pos = heating_start_z + (turn / total_turns) * (heating_end_z - heating_start_z)
            x_pos = coil_radius * np.cos(angle)
            y_pos = coil_radius * np.sin(angle)
            ax.scatter(x_pos, y_pos, z_pos, c='darkred', s=40, alpha=0.9)
            
            # Add turn number annotation for key turns only
            if i == 0:
                ax.text(x_pos, y_pos, z_pos, f'Turn 0', fontsize=8, color='darkred', fontweight='bold')
            elif i == len(turn_positions) - 1:
                ax.text(x_pos, y_pos, z_pos, f'Turn {total_turns}', fontsize=8, color='darkred', fontweight='bold')
    
    # ==================== TRANSITION INTERFACE ====================
    print("4. Showing cylindrical-to-cubic transition...")
    
    # Mark transition boundary (reflective aluminum outer surface)
    r_transition = mesh.cylindrical_outer_radius
    X_trans = r_transition * np.cos(theta)
    Y_trans = r_transition * np.sin(theta)
    
    # Plot transition circles at heating zone boundaries ONLY (not cold zones)
    z_positions = [config.HEATING_COIL_START, config.HEATING_COIL_END]  # Start and end of heating zone
    for z_pos in z_positions:
        Z_trans = np.full_like(theta, z_pos)
        ax.plot(X_trans, Y_trans, Z_trans, 'red', linewidth=3, alpha=0.8)
    
    # Add one circle in middle of heating zone for reference
    z_middle_heating = (config.HEATING_COIL_START + config.HEATING_COIL_END) / 2
    Z_trans_mid = np.full_like(theta, z_middle_heating)
    ax.plot(X_trans, Y_trans, Z_trans_mid, 'red', linewidth=2, alpha=0.6)
    
    # ==================== CUBIC REGION BOUNDARIES ====================
    print("5. Drawing cubic enclosure boundaries (axisymmetric view)...")
    
    if mesh.x_nodes is not None:
        # Draw full aluminum box outline - complete 3D cube
        box_size = mesh.aluminum_box_outer
        # Full cube corners
        box_corners = [
            [-box_size, -box_size], [box_size, -box_size], 
            [box_size, box_size], [-box_size, box_size], [-box_size, -box_size]
        ]
        
        # Draw box outline at key z positions (full 3D view)
        for z_pos in [0, config.FURNACE_LENGTH]:
            for i in range(len(box_corners)-1):
                x1, y1 = box_corners[i]
                x2, y2 = box_corners[i+1] 
                ax.plot([x1, x2], [y1, y2], [z_pos, z_pos], 'k-', linewidth=2, alpha=0.8)
        
        # Draw vertical edges (full cube)
        for x, y in box_corners[:-1]:  # Exclude duplicate last corner
            ax.plot([x, x], [y, y], [0, config.FURNACE_LENGTH], 'k-', linewidth=1, alpha=0.6)
    
    # Mesh node visualization removed per user request
    
    # ==================== FORMATTING AND LABELS ====================
    
    # Set equal aspect ratio and labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set limits for better view
    max_extent = mesh.aluminum_box_outer * 1.1
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([0, config.FURNACE_LENGTH])
    
    # Add title to 3D plot
    ax.set_title('3D Tube Furnace Mesh Visualization', fontsize=16, fontweight='bold', pad=20)
    
    # ==================== LEGEND AND INFO PANEL ====================
    
    # Calculate mesh statistics
    cylindrical_nodes = len(mesh.r_nodes) * len(mesh.z_nodes)
    cubic_nodes = len(mesh.cubic_nodes_xyz) if hasattr(mesh, 'cubic_nodes_xyz') else 0
    total_nodes = cylindrical_nodes + cubic_nodes
    
    # Create organized information panel
    y_pos = 0.95
    line_spacing = 0.04
    
    # Title for info panel
    info_ax.text(0.02, y_pos, 'FURNACE MESH ANALYSIS', fontsize=16, fontweight='bold', 
                transform=info_ax.transAxes)
    y_pos -= line_spacing * 1.5
    
    # Physical Structure Section
    info_ax.text(0.02, y_pos, 'PHYSICAL STRUCTURE (inner → outer):', fontsize=12, fontweight='bold', 
                transform=info_ax.transAxes, color='navy')
    y_pos -= line_spacing
    
    # Use exact color codes from cross_section_generator.py
    material_colors = {
        'sample_space': '#87CEEB',      # Sky blue - Sample
        'quartz_glass': '#D3D3D3', # Light gray - Glass
        'imperial_cement': '#8B4513',    # Saddle brown - Cement
        'lyrufexon_ceramic': '#FFD700',  # Gold - Ceramic
        'reflective_aluminum': '#C0C0C0', # Silver - Reflective
        'air_gap': '#E0FFFF',           # Light cyan - Air Gap
        'aluminum_5052': '#696969',     # Dim gray - Al_5052
        'heating_coil': '#FF0000'       # Red for heating coil
    }
    
    structure_items = [
        ('Quartz glass', material_colors['quartz_glass']),
        ('Heating element', material_colors['heating_coil']),
        ('Furnace cement', material_colors['imperial_cement']),
        ('Ceramic wool', material_colors['lyrufexon_ceramic']),
        ('Reflective aluminum', material_colors['reflective_aluminum']),
        ('Air gap', material_colors['air_gap']),
        ('Aluminum box', material_colors['aluminum_5052'])
    ]
    
    for item, color in structure_items:
        # Add background for better readability
        info_ax.text(0.05, y_pos, f'• {item}', fontsize=10, transform=info_ax.transAxes, 
                    color='white' if color in ['#8B4513', '#696969', '#FF0000'] else 'black', 
                    fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8, edgecolor='black'))
        y_pos -= line_spacing * 0.8
    
    y_pos -= line_spacing * 0.5
    
    # Heating Zone Section (moved to replace removed physical structure items)
    info_ax.text(0.35, y_pos + line_spacing * 6, 'HEATING ZONE:', fontsize=12, fontweight='bold', 
                transform=info_ax.transAxes, color='darkred')
    heating_y = y_pos + line_spacing * 5
    info_ax.text(0.38, heating_y, f'Power: {config.HEATING_COIL_POWER:.0f}W', 
            fontsize=9, transform=info_ax.transAxes, color='red')
    heating_y -= line_spacing * 0.8
    info_ax.text(0.38, heating_y, f'Length: {config.HEATING_COIL_LENGTH*1000:.1f}mm (10")', 
                fontsize=9, transform=info_ax.transAxes, color='red')
    heating_y -= line_spacing * 0.8
    info_ax.text(0.38, heating_y, f'Turns: {config.HEATING_COIL_TURNS} (helix visualization)', 
                fontsize=9, transform=info_ax.transAxes, color='red')
    heating_y -= line_spacing * 0.8
    info_ax.text(0.38, heating_y, f'Position: {config.HEATING_COIL_START*1000:.1f} → {config.HEATING_COIL_END*1000:.1f}mm', 
                fontsize=9, transform=info_ax.transAxes, color='red')
    heating_y -= line_spacing * 0.8
    info_ax.text(0.38, heating_y, 'Red lines: 10" heating zone', 
                fontsize=9, transform=info_ax.transAxes, color='red', fontweight='bold')
    heating_y -= line_spacing * 0.8
    info_ax.text(0.38, heating_y, 'Red circles: Cylindrical-cubic transition', 
                fontsize=9, transform=info_ax.transAxes, color='red')
    
    y_pos -= line_spacing * 1.2
    
    # Mesh Statistics Section
    info_ax.text(0.02, y_pos, 'MESH STATISTICS:', fontsize=12, fontweight='bold', 
                transform=info_ax.transAxes, color='darkgreen')
    y_pos -= line_spacing
    info_ax.text(0.05, y_pos, f'Cylindrical nodes: {cylindrical_nodes:,}', 
                fontsize=11, transform=info_ax.transAxes, fontweight='bold')
    y_pos -= line_spacing * 0.8
    info_ax.text(0.05, y_pos, f'Cubic nodes: {cubic_nodes:,}', 
                fontsize=11, transform=info_ax.transAxes, fontweight='bold')
    y_pos -= line_spacing * 0.8
    info_ax.text(0.05, y_pos, f'Total nodes: {total_nodes:,}', 
                fontsize=12, transform=info_ax.transAxes, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    y_pos -= line_spacing * 1.2
    
    # Technical Details Section
    info_ax.text(0.02, y_pos, 'TECHNICAL DETAILS:', fontsize=12, fontweight='bold', 
                transform=info_ax.transAxes, color='purple')
    y_pos -= line_spacing
    info_ax.text(0.05, y_pos, f'Cylindrical extent: 0 → {mesh.cylindrical_outer_radius*1000:.1f}mm radius', 
                fontsize=10, transform=info_ax.transAxes)
    y_pos -= line_spacing * 0.8
    
    if hasattr(mesh, 'cubic_nodes_xyz') and mesh.cubic_nodes_xyz is not None:
        # Elaborate on cubic region dimensions
        cubic_start = mesh.cylindrical_outer_radius * 1000  # Where cubic region starts
        aluminum_outer = mesh.aluminum_box_outer * 1000  # Aluminum box outer boundary
        
        info_ax.text(0.05, y_pos, f'Cubic extent: {cubic_start:.1f}mm → {aluminum_outer:.1f}mm square', 
                    fontsize=10, transform=info_ax.transAxes)
        y_pos -= line_spacing * 0.8
        
        # Air gap and aluminum wall details
        air_gap_outer = mesh.air_gap_outer * 1000  # Air gap outer boundary
        air_gap_thickness = air_gap_outer - cubic_start
        aluminum_thickness = aluminum_outer - air_gap_outer
        
        info_ax.text(0.05, y_pos, f'Air gap thickness: {air_gap_thickness:.1f}mm', 
                    fontsize=9, transform=info_ax.transAxes, color='darkblue', fontweight='bold')
        y_pos -= line_spacing * 0.7
        info_ax.text(0.05, y_pos, f'Aluminum thickness: {aluminum_thickness:.1f}mm', 
                    fontsize=9, transform=info_ax.transAxes, color='darkred', fontweight='bold')
    
    y_pos -= line_spacing * 1.2
    
    y_pos -= line_spacing * 0.5
       
    # Adjust viewing angle for full 3D perspective
    ax.view_init(elev=20, azim=45)  # Standard 3D view to show full array structure
    
    # Adjust layout to give more space to 3D plot
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.1)
    
    # Show the plot
    plt.show()
    
    print("\n3D mesh visualization complete!")
    
    # ==================== MESH ANALYSIS ====================
    
    print("\n" + "="*60)
    print("MESH ANALYSIS")
    print("="*60)
    
    print(f"\nCYLINDRICAL REGION:")
    print(f"  Radial layers: {len(mesh.r_nodes)} nodes")
    print(f"  Axial sections: {len(mesh.z_nodes)} nodes")  
    print(f"  Physical extent: 0 → {mesh.reflective_outer_radius*1000:.1f}mm radius")
    print(f"  Total length: {config.FURNACE_LENGTH*1000:.1f}mm (12 inches)")
    
    if hasattr(mesh, 'cubic_nodes_xyz') and mesh.cubic_nodes_xyz is not None:
        print(f"\nCUBIC REGION:")
        print(f"  Valid coordinate pairs: {len(mesh.cubic_nodes_xyz) // len(mesh.z_nodes_cubic)}")
        print(f"  Z sections: {len(mesh.z_nodes_cubic)} nodes") 
        print(f"  Total cubic nodes: {len(mesh.cubic_nodes_xyz):,}")
        print(f"  Physical extent: ±{mesh.aluminum_box_outer*1000:.1f}mm square cross-section")
        print(f"  Minimum distance from center: {min(np.sqrt(x**2 + y**2) for x, y, z in mesh.cubic_nodes_xyz)*1000:.1f}mm")
        print(f"  Air gap: {mesh.cylindrical_outer_radius*1000:.1f} → {mesh.air_gap_outer*1000:.1f}mm")
        print(f"  Aluminum walls: {mesh.air_gap_outer*1000:.1f} → {mesh.aluminum_box_outer*1000:.1f}mm")
    
    # Calculate and display totals
    total_cylindrical = len(mesh.r_nodes) * len(mesh.z_nodes)
    total_cubic = len(mesh.cubic_nodes_xyz) if hasattr(mesh, 'cubic_nodes_xyz') else 0
    total_all = total_cylindrical + total_cubic
    
    print(f"\nHEATING ZONE:")
    print(f"  Position: {config.HEATING_COIL_START*1000:.1f} → {config.HEATING_COIL_END*1000:.1f}mm")
    print(f"  Length: {config.HEATING_COIL_LENGTH*1000:.1f}mm (10 inches)")
    print(f"  Coverage: {(config.HEATING_COIL_LENGTH/config.FURNACE_LENGTH)*100:.1f}% of total length")
    
    print(f"\nTRANSITION INTERFACE:")
    print(f"  Cylindrical boundary: {mesh.cylindrical_outer_radius*1000:.1f}mm radius")
    print(f"  Transition to cubic at reflective aluminum outer surface")
    
    print(f"\nMESH SUMMARY:")
    print(f"  Cylindrical nodes: {total_cylindrical:,}")
    print(f"  Cubic nodes: {total_cubic:,}")
    print(f"  Total nodes: {total_all:,}")
    print(f"  Representation: Sparse cubic (filtered coordinates)")
    
    return fig, ax

if __name__ == "__main__":
    # Run the visualization
    fig, ax = visualize_3d_mesh()
    
    print("\nVisualization saved to display. Close the plot window to continue.")