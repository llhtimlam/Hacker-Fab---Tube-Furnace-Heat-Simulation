"""
Correct Tube Furnace Mesh Generator - Rewritten from Scratch
Following exact physical structure from inner to outer
"""

import numpy as np
import matplotlib.pyplot as plt
import config

class TubeFurnaceMesh:
    """
    Tube Furnace Mesh following correct physical structure:
    
    CYLINDRICAL REGION (inner components):
    1. Sample region (centerline) 
    2. Quartz glass tube
    3. Heating element  
    4. Furnace cement
    5. Ceramic wool
    6. Reflective aluminium
    
    TRANSITION: Reflective aluminium outer surface
    
    CUBIC REGION (outer enclosure):
    7. Air gap
    8. Aluminium box
    """
    
    def __init__(self):
        print("Creating correct tube furnace mesh from scratch")
        self.r_nodes = None
        self.z_nodes = None

        # Physical boundaries (from config)
        self.sample_radius = config.GLASS_TUBE_INNER_RADIUS
        self.glass_outer_radius = config.GLASS_TUBE_OUTER_RADIUS
        self.kanthal_outer_radius = config.HEATING_COIL_RADIUS
        self.cement_outer_radius = config.GLASS_TUBE_OUTER_RADIUS + config.FURNACE_CEMENT_THICKNESS
        self.ceramic_outer_radius = config.INSULATION_OUTER_RADIUS
        self.reflective_outer_radius = config.REFLECTIVE_CASING_OUTER_RADIUS

        self.kanthal_axial_start = config.HEATING_COIL_START
        self.kanthal_axial_end = config.HEATING_COIL_END
        
        self.sample_glass_interface = None
        self.glass_kanthal_cement_interface = None
        self.kanthal_cement_interface = None
        self.cement_ceramic_interface = None
        self.ceramic_reflective_interface = None
        self.reflective_outer_interface = None

        self.cold_heat_interface_front = None
        self.cold_heat_interface_back = None

        self.kanthal_r_position_start = None
        self.kanthal_r_position_end = None
        self.kanthal_z_position_start = None
        self.kanthal_z_position_end = None

        self.inner_glass_tube_area = None
        self.cubic_inner_surface = None
        self.cubic_outer_surface = None
        
        # Cubic Volumes for energy calculations
        self.cubic_air_gap_volume = None
        self.cubic_aluminium_casing_volume = None
        
        # Cylindrical region boundary: outer surface of reflective aluminum foil
        self.cylindrical_outer_radius = self.reflective_outer_radius
        self.cylindrical_length = config.FURNACE_LENGTH  # 12 inches total length

        # Cubic region boundary: outer surface of aluminium casing
        self.cubic_inner_dimension = config.ENCLOSURE_INNER_SIZE
        self.cubic_outer_dimension = config.ENCLOSURE_OUTER_SIZE
        self.cubic_depth = config.ENCLOSURE_WALL_THICKNESS
        self.cubic_distance = config.AIR_GAP_THICKNESS

        # Mesh arrays (will be populated)
        self.cylindrical_mesh = None

        self.r_centers = None
        self.z_centers = None
        self.r_faces = None  # Internal variable for faces
        self.z_faces = None  # Internal variable for faces
        self.generate_complete_mesh()
        
    def generate_complete_mesh(self):
        """Generate the complete mesh following physical structure"""
        print("\n=== GENERATING CORRECT TUBE FURNACE MESH ===")
        
        # Step 1: Generate cylindrical region (inner components)
        print("1. Generating CYLINDRICAL region (sample -> reflective aluminium)")
        self._generate_cylindrical_region()
               
        # Step 2: Create material mapping
        print("2. Creating material mapping")
        self._create_material_mapping()
        
        print("=== MESH GENERATION COMPLETE ===\n")
        self._print_mesh_summary()
        
    def _generate_cylindrical_region(self):
        """Generate cylindrical mesh for inner components"""
        
        # Radial discretization Negatively Skewed, positively centered
        # For example: | Faces o Centered Nodes (Array Index, @ coordinate)
        # |(0,@0) o(0,@0.5) |(1,@1) o(1,@1.25)|(2,@1.5) o(2,2.75) |(3,@2) ... etc
        r_node = []
        
        # Layer 1: Sample region (0 to sample_radius)
        #r_sample = np.linspace(0, self.sample_radius, config.RADIAL_NODES_SAMPLE, endpoint=True)
        #r_nodes.extend(r_sample)  # Placeholder at center point of sample region  (single node)

        # Layer 2: Glass tube (sample_radius to glass_outer_radius)  
        r_glass = np.linspace(self.sample_radius, self.glass_outer_radius, config.RADIAL_NODES_GLASS, endpoint=True)[1:]
        r_node.extend(r_glass) # Exclude overlap with next layer
        
        # Layer 2: Furnace cement AND Kanthal heating element (glass_outer to kanthal_outer)
        r_kanthal_cement = np.linspace(self.glass_outer_radius, self.kanthal_outer_radius, config.RADIAL_NODES_KANTHAL, endpoint=True)[1:]
        r_node.extend(r_kanthal_cement) # Required at least 2 point to find center

        # Layer 3: Furnace cement (glass_outer to cement_outer)
        r_cement = np.linspace(self.kanthal_outer_radius, self.cement_outer_radius, config.RADIAL_NODES_CEMENT, endpoint=True)[1:]
        r_node.extend(r_cement) # Exclude overlap with next and previous layer
        
        # Layer 4: Ceramic wool (cement_outer to ceramic_outer)
        r_ceramic = np.linspace(self.cement_outer_radius, self.ceramic_outer_radius, config.RADIAL_NODES_CERAMIC, endpoint=True)[1:]
        r_node.extend(r_ceramic) # Exclude overlap with next layer
        
        # Layer 5: Reflective aluminium (ceramic_outer to reflective_outer)
        r_reflective = np.linspace(self.ceramic_outer_radius, self.reflective_outer_radius, config.RADIAL_NODES_REFLECTIVE, endpoint=True)
        r_node.extend(r_reflective) # Required at least 2 point to find center
        
        self.r_nodes = np.unique(r_node)
        
        # Axial discretization Negatively Skewed, positively centered
        # Heating region override cold zone
        # Axial Facial Node Index Example: Cold 2 Heating 3 Cold 2
        # |0 |1 |2 |3 |4 |5 |6     # |2 |4 got replaced to Heating
        # |0-1|2-4|5-6|            # | Cold | Heating | Cold |
        # Heating Centered Node is used to for heating coil mapping
        z_node = []
        
        # COLD ZONE 1: Before heating (1 inch with nodes for discretization)
        z_before = np.linspace(0, config.HEATING_COIL_START, config.AXIAL_NODES_BEFORE + 1, endpoint=True)  # +1 to include boundary
        z_node.extend(z_before)  # Exclude overlap with heating end

        # HEATING ZONE: Heating region (CRITICAL: exactly 39 nodes for 39 coil turns)
        z_heating = np.linspace(config.HEATING_COIL_START, config.HEATING_COIL_END, config.AXIAL_NODES_HEATING, endpoint=True)[1:]
        z_node.extend(z_heating)  # Add all heating nodes

        # COLD ZONE 2: After heating (1 inch with nodes for discretization)
        z_after = np.linspace(config.HEATING_COIL_END, config.FURNACE_LENGTH, config.AXIAL_NODES_AFTER + 1, endpoint=True)[1:]  # +1 to include boundary
        z_node.extend(z_after)  # Exclude overlap with heating end

        self.z_nodes = np.unique(z_node)

        # Create cylindrical meshgrid
        #self.sample_glass_interface = np.searchsorted(self.r_nodes, self.sample_radius)
        # Convert to faces and centers for solver.py for Finite Volume Method (FVM) Lumped-Element Model Hybrid Heat Simulation System
        self.r_faces = self.r_nodes # Trimmed Sample Air Space Region for Cylindrical Continuous Conduction Region for FVM
        self.z_faces = self.z_nodes
        # Cell centers are the average of adjacent faces
        self.r_centers = (self.r_faces[:-1] + self.r_faces[1:]) / 2.0
        self.z_centers = (self.z_faces[1:] + self.z_faces[:-1]) / 2.0

        # Radial interface indices
        self.glass_kanthal_cement_interface = np.searchsorted(self.r_faces, self.glass_outer_radius)
        self.kanthal_cement_interface = np.searchsorted(self.r_faces, self.kanthal_outer_radius)
        self.cement_ceramic_interface = np.searchsorted(self.r_faces, self.cement_outer_radius)
        self.ceramic_reflective_interface = np.searchsorted(self.r_faces, self.ceramic_outer_radius)
        self.reflective_outer_interface = np.searchsorted(self.r_faces, self.reflective_outer_radius)  # Last valid index

        # Axial interface indices
        self.cold_heat_interface_front = np.searchsorted(self.z_faces, config.HEATING_COIL_START)
        self.cold_heat_interface_back = np.searchsorted(self.z_faces, config.HEATING_COIL_END)

        # Kanthal coil position based on coordinates, not node counts
        self.kanthal_r_position_start = np.searchsorted(self.r_faces, self.glass_outer_radius)
        self.kanthal_r_position_end = np.searchsorted(self.r_faces, self.kanthal_outer_radius)
        self.kanthal_z_position_start = np.searchsorted(self.z_faces, config.HEATING_COIL_START)
        self.kanthal_z_position_end = np.searchsorted(self.z_faces, config.HEATING_COIL_END)

        # Cylindrical surface boundaries
        self.inner_glass_tube_area = 2 * np.pi * self.sample_radius * config.FURNACE_LENGTH

        # Cylindrical Volumes for energy calculations
        self.sample_air_space_volume = np.pi * self.sample_radius**2 *  self.cylindrical_length

        # Cubic surface boundaries
        self.cubic_inner_surface = 4 * self.cubic_inner_dimension ** 2 * self.cylindrical_length
        self.cubic_outer_surface = 4 * self.cubic_outer_dimension ** 2 * self.cylindrical_length
        
        # Cubic Volumes for energy calculations
        self.cubic_air_gap_volume = (self.cubic_inner_dimension**2 * self.cylindrical_length) - (np.pi * (self.cylindrical_outer_radius**2) * self.cylindrical_length) # Assume 2D cylindrical system
        self.cubic_aluminium_casing_volume = (self.cubic_outer_dimension**2 - self.cubic_inner_dimension**2) * self.cylindrical_length
        
        # Axial discretization for 12-inch total length with proper cold/heating zones
        cold_zone_length_inches = (config.FURNACE_LENGTH - config.HEATING_COIL_LENGTH) / 2 / 0.0254  # Convert to inches
        heating_zone_length_inches = config.HEATING_COIL_LENGTH / 0.0254
        # Verify cold zone dimensions and node density
        cold_zone_length_mm = (config.FURNACE_LENGTH - config.HEATING_COIL_LENGTH) / 2 * 1000
        cold_zone_node_spacing = cold_zone_length_mm / config.AXIAL_NODES_BEFORE
        # Calculate actual node spacing for validation
        cold_zone_spacing_mm = (config.HEATING_COIL_START * 1000) / config.AXIAL_NODES_BEFORE
        heating_zone_spacing_mm = (config.HEATING_COIL_LENGTH * 1000) / config.AXIAL_NODES_HEATING
        print(f"   Total furnace length: {config.FURNACE_LENGTH*1000:.1f}mm ({config.FURNACE_LENGTH/0.0254:.1f} inches)")
        print(f"   Heating coil length: {config.HEATING_COIL_LENGTH*1000:.1f}mm ({heating_zone_length_inches:.1f} inches)")
        print(f"   Cold zones (each): {cold_zone_length_inches:.1f} inches = {cold_zone_length_mm:.1f}mm")
        print(f"   Cold zone node spacing: {cold_zone_node_spacing:.2f}mm/node ({config.AXIAL_NODES_BEFORE} nodes per cold zone)")
        print(f"   Coil position: {config.HEATING_COIL_START*1000:.1f}mm to {config.HEATING_COIL_END*1000:.1f}mm")        
        print(f"   CRITICAL VALIDATION - Heating coil mapping:")
        print(f"     - Coil turns: {config.HEATING_COIL_TURNS}")
        print(f"     - Heating nodes: {config.AXIAL_NODES_HEATING}")
        print(f"     - Mapping ratio: {config.HEATING_COIL_TURNS/config.AXIAL_NODES_HEATING:.3f} (MUST = 1.000)")
        print(f"   Zone breakdown:")
        print(f"     - Cold zone 1: 0 to {config.HEATING_COIL_START*1000:.1f}mm ({config.AXIAL_NODES_BEFORE} nodes, {cold_zone_spacing_mm:.2f}mm/node)")
        print(f"     - Heating zone: {config.HEATING_COIL_START*1000:.1f} to {config.HEATING_COIL_END*1000:.1f}mm ({config.AXIAL_NODES_HEATING} nodes, {heating_zone_spacing_mm:.2f}mm/node)")  
        print(f"     - Cold zone 2: {config.HEATING_COIL_END*1000:.1f} to {config.FURNACE_LENGTH*1000:.1f}mm ({config.AXIAL_NODES_AFTER} nodes, {cold_zone_spacing_mm:.2f}mm/node)")
        print(f"   Cylindrical mesh: {len(self.r_nodes)} radial x {len(self.z_nodes)} axial = {len(self.r_nodes) * len(self.z_nodes):,} nodes")
        print(f"   Radial range: 0 to {self.reflective_outer_radius*1000:.1f}mm")
        print(f"   Axial range: 0 to {config.FURNACE_LENGTH*1000:.1f}mm (12 inches total)")
        print(f"   Heating zone: {config.HEATING_COIL_START*1000:.1f} to {config.HEATING_COIL_END*1000:.1f}mm (10 inches)")

    def _create_material_mapping(self):
        """Create material mapping for both regions"""
        
        # CYLINDRICAL REGION materials (following physical structure)
        #self.material_map_cylindrical_faced = np.full((len(self.r_faces), len(self.z_faces)), -1, dtype=int)
        self.material_map_cylindrical_centered = np.full((len(self.r_centers), len(self.z_centers)), -1, dtype=int)
        # Use meshgrid to create a 2D grid of r and z coordinates for boolean indexing
        r_grid, z_grid = np.meshgrid(self.r_centers, self.z_centers, indexing='ij')
        # Use 2D boolean indexing for fast material assignment
        # The order of these assignments is important; start from the innermost layer
        # and work your way outwards to avoid overwriting.
        
        # Layer 0: Sample region (centerline)
        mask = r_grid <= self.sample_radius
        self.material_map_cylindrical_centered[mask] = 0

        # Layer 1: Quartz glass
        mask = (r_grid > self.sample_radius) & (r_grid <= self.glass_outer_radius)
        self.material_map_cylindrical_centered[mask] = 1

        # Layer 3: Furnace cement
        mask = (r_grid > self.glass_outer_radius) & (r_grid <= self.cement_outer_radius)
        self.material_map_cylindrical_centered[mask] = 3

        # Layer 4: Ceramic wool
        mask = (r_grid > self.cement_outer_radius) & (r_grid <= self.ceramic_outer_radius)
        self.material_map_cylindrical_centered[mask] = 4

        # Layer 5: Reflective aluminium
        mask = (r_grid > self.ceramic_outer_radius) & (r_grid <= self.reflective_outer_radius)
        self.material_map_cylindrical_centered[mask] = 5

        # Kanthal heating element (overwrites furnace cement in a specific region)
        kanthal_indices = (
            (r_grid >= self.glass_outer_radius) & (r_grid <= self.kanthal_outer_radius) &
            (z_grid >= self.kanthal_axial_start) & (z_grid <= self.kanthal_axial_end)
        )
        self.material_map_cylindrical_centered[kanthal_indices] = 2

        # Verify that no nodes were missed (e.g., still have -1)
        if (self.material_map_cylindrical_centered == -1).any():
            print("Warning: Some mesh nodes were not assigned a material.")
    
    def _print_mesh_summary(self):
        """Print comprehensive mesh summary"""
        
        print("MESH SUMMARY:")
        print("="*60)
        
        print("\nCYLINDRICAL REGION (inner components):")
        print(f"  Physical layers: Sample -> Glass -> Cement -> Ceramic -> Reflective Al")
        print(f"  Radial nodes: {len(self.r_nodes)}")
        print(f"  Axial nodes: {len(self.z_nodes)}")  
        print(f"  Total cylindrical nodes: {len(self.r_nodes) * len(self.z_nodes):,}")
        print(f"  Radial extent: 0 to {self.reflective_outer_radius*1000:.1f}mm")
        print("  Axial structure (12-inch total with cold zones):")
        print(f"    - Cold zone 1: 0 to {config.HEATING_COIL_START*1000:.1f}mm (axial heat transfer)")
        print(f"    - Heating zone: {config.HEATING_COIL_START*1000:.1f} to {config.HEATING_COIL_END*1000:.1f}mm (direct heating)")  
        print(f"    - Cold zone 2: {config.HEATING_COIL_END*1000:.1f} to {config.FURNACE_LENGTH*1000:.1f}mm (axial heat transfer)")
        print(f"    - Total length: {config.FURNACE_LENGTH*1000:.1f}mm (12 inches)")
        print(f"\nTOTAL MESH NODES: {len(self.r_nodes) * len(self.z_nodes):,}")
        print("="*60)

# Test the new mesh
if __name__ == "__main__":
    mesh = TubeFurnaceMesh()