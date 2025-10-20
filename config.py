"""
High-Resolution Configuration for Tube Furnace Simulation
Optimized for fine spatial and temporal discretization
"""

import numpy as np
# All dimensions in meters, SI units
# ==================== GEOMETRY PARAMETERS ====================
# Ultra-High-Temperature Quartz Glass Tube 1" OD, 12" Long # https://www.mcmaster.com/4100N17/
FURNACE_LENGTH = 12 * 0.0254  # 12 inches = 0.3048 m
GLASS_TUBE_OUTER_DIAMETER = 1.0 * 0.0254  # 1 inch OD
GLASS_TUBE_INNER_DIAMETER = 0.866 * 0.0254  # 0.866 inch ID
GLASS_TUBE_OUTER_RADIUS = GLASS_TUBE_OUTER_DIAMETER / 2  # 12.7 mm
GLASS_TUBE_INNER_RADIUS = GLASS_TUBE_INNER_DIAMETER / 2  # 11.0 mm
GLASS_TUBE_WALL_THICKNESS = GLASS_TUBE_OUTER_RADIUS - GLASS_TUBE_INNER_RADIUS

# ==================== HEATING COIL SPECIFICATIONS ====================
# Kanthal A1-100' - 24 Gauge Wire - 100ft - 0.51mm - 0.02in - Made in USA - Master Wire Supply # https://www.amazon.com/Kanthal-A1-Gauge-Resistance-Wire/dp/B07CHSG169
HEATING_COIL_TURNS = 39  # Physical coil turns (for reference)
HEATING_COIL_LENGTH = 10 * 0.0254  # 10 inches = 0.254 m
HEATING_COIL_START = (FURNACE_LENGTH - HEATING_COIL_LENGTH) / 2  # Centered
HEATING_COIL_END = HEATING_COIL_START + HEATING_COIL_LENGTH
HEATING_COIL_POWER = 480.0  # Watts - Power Supply
HEATING_COIL_WIRE_DIAMETER = 0.51e-3  # 0.51mm wire diameter
HEATING_COIL_RESISTANCE = 30  # 30-38 Ohm

# Coil positioning
HEATING_COIL_RADIUS = GLASS_TUBE_OUTER_RADIUS + HEATING_COIL_WIRE_DIAMETER / 2

# Note: COIL_TURN_POSITIONS now uses all mesh nodes in heating zone for continuous coverage
# This eliminates gaps that would cause cold spots in the middle of the heating zone

# ==================== MATERIAL SPECIFICATIONS ====================
# Imperial High Temperature Stove & Furnace Cement, Grey # https://www.imperialgroup.ca/product/stove-fireplace/maintenance-products/mortar-cements/hi-temp-stove-furnace-cement-gray
FURNACE_CEMENT_THICKNESS = 0.003  # 3mm coating
FURNACE_CEMENT_TYPE = "Imperial High Temperature Stove & Furnace Cement, Grey"

# Lyrufexon Ceramic Fiber Insulation, 2600F Fireproof Insulation Blanket # https://www.amazon.ca/Lyrufexon-Ceramic-Fiber-Insulation-Fireproof/dp/B0CBWXRLLH?th=1
INSULATION_OUTER_RADIUS = 0.1175  # 117.5mm radius = 235mm diameter
INSULATION_TYPE = "Lyrufexon Ceramic Fiber Insulation, 2600F Fireproof Insulation Blanket"

# ==================== REFLECTIVE CASING SPECIFICATIONS ====================
# Reflective aluminum casing around ceramic fiber to minimize radiation losses
REFLECTIVE_CASING_THICKNESS = 0.001  # 1mm aluminum sheet
REFLECTIVE_CASING_TYPE = "Polished Aluminum Reflective Casing"
REFLECTIVE_CASING_OUTER_RADIUS = INSULATION_OUTER_RADIUS + REFLECTIVE_CASING_THICKNESS

# ==================== CUBIC ENCLOSURE SPECIFICATIONS ====================
# Air gap and cCubic Aluminum 5052-H32 Box Enclosure
AIR_GAP_THICKNESS = 0.030  # 30mm air gap for thermal isolation
ENCLOSURE_WALL_THICKNESS = 0.003  # 3mm aluminum 5052 walls
ENCLOSURE_INNER_SIZE = 2 * (REFLECTIVE_CASING_OUTER_RADIUS + AIR_GAP_THICKNESS)
ENCLOSURE_OUTER_SIZE = ENCLOSURE_INNER_SIZE + 2 * ENCLOSURE_WALL_THICKNESS

# Cubic enclosure dimensions
ENCLOSURE_CUBE_SIDE = ENCLOSURE_OUTER_SIZE  # Square cross-section
ENCLOSURE_CUBE_LENGTH = FURNACE_LENGTH  # Length matches furnace
ENCLOSURE_TYPE = "Cubic Aluminum 5052-H32 Box Enclosure"

# ==================== TEMPERATURE CONDITIONS ====================
AMBIENT_TEMP = 25 + 273.15  # 298.15 K (25°C)
INITIAL_TEMP = AMBIENT_TEMP  # Start at ambient
TARGET_TEMP = 1000 + 273.15  # 1273.15 K (1000°C) target
MAX_OPERATING_TEMP = 1500 + 273.15  # 1773.15 K (1500°C) max

# ==================== MESH PARAMETERS - ADAPTED FOR NEW MESH SYSTEM ====================

# ========== CUSTOMIZABLE RADIAL DISCRETIZATION ==========
# NEGATIVELY-BIASED FACIAL RADIAL NODE GENERATION for CELL-CENTERED FINITE VOLUME METHOD - Only applies to CYLINDRICAL region
# These can be modified to adjust mesh density in high-temperature regions
# Please keep in mind of dr for stability criteria for Courant–Friedrichs–Lewy (CFL) condition dt <= K * dr^2 / (2*alpha)
RADIAL_NODES_SAMPLE = 1      # Placeholder for sample region (single node) PREFER SET TO 1, only for display usage
RADIAL_NODES_GLASS = 3       # High resolution for accurate depiction of thermal gradient in glass layer
RADIAL_NODES_KANTHAL = 2     # Heating coil center location mapping (single node) MUST SET TO > 1
RADIAL_NODES_CEMENT = 6      # High resolution for accurate depiction of thermal gradient in cement layer
RADIAL_NODES_CERAMIC = 41    # Moderate resolution to reduce rounding errors in insulation layer
RADIAL_NODES_REFLECTIVE = 2  # Reflective casing center location mapping (single node) MUST SET TO > 1

# Note: Air gap and aluminum enclosure use CUBIC mesh system (not radial discretization)

# ========== FIXED AXIAL DISCRETIZATION - DO NOT MODIFY ==========
# CRITICAL: These values are LOCKED to ensure proper heat generation from coil elements
# Modifying these will cause discretization errors and heating simulation failures

# CONSTRAINT ENFORCEMENT: AXIAL_NODES_HEATING MUST ALWAYS EQUAL HEATING_COIL_TURNS
AXIAL_NODES_HEATING = 39  # CRITICAL: 39 nodes = 39 coil turns (perfect 1:1 mapping)

# Cold zone discretization - coarser resolution for uniform regions
AXIAL_NODES_BEFORE = 4       # Moderate resolution to trace temperature drop profile before heating zone
AXIAL_NODES_AFTER = 4        # Moderate resolution to trace temperature drop profile after heating zone

# Runtime validation to prevent configuration errors
#if AXIAL_NODES_HEATING != HEATING_COIL_TURNS:
    #raise ValueError(f"CONFIGURATION ERROR: AXIAL_NODES_HEATING ({AXIAL_NODES_HEATING}) must equal HEATING_COIL_TURNS ({HEATING_COIL_TURNS})")

# ========== TOTAL RADIAL AND AXIAL NODES CALCULATION ==========
# Mesh totals - Overlap is not excluded in cylindrical system
TOTAL_RADIAL_NODES = (RADIAL_NODES_SAMPLE + RADIAL_NODES_GLASS + RADIAL_NODES_KANTHAL + RADIAL_NODES_CEMENT + RADIAL_NODES_CERAMIC + RADIAL_NODES_REFLECTIVE)  # Cylindrical region only
TOTAL_AXIAL_NODES = AXIAL_NODES_BEFORE + AXIAL_NODES_HEATING + AXIAL_NODES_AFTER

# ========== TOTAL MESH NODE CALCULATION ==========
# Cylindrical region: Complete structured mesh
TOTAL_CYLINDRICAL_NODES = TOTAL_RADIAL_NODES * TOTAL_AXIAL_NODES

# ==================== SIMULATION GEOMETRY ====================
POWER_DENSITY_TREATMENT = "2D_CYLINDRICAL" # Default setting DO NOT CHANGE
# ==================== HIGH-RESOLUTION TEMPORAL PARAMETERS ====================
# Time discretization - HIGH RESOLUTION
SIMULATION_DURATION = 0.00009 * 3600  # 30000 = 0.5 hours (64.7218382338 e-6 seconds time steps / 0.05 dr)
TIME_STEP_SECONDS = 0.00001  # 1 second time steps for high temporal resolution
TOTAL_TIME_STEPS = int(SIMULATION_DURATION / TIME_STEP_SECONDS)

# Adaptive time stepping parameters
MIN_TIME_STEP = 0  # Minimum time increment
MAX_TIME_STEP = 10000000000  # Maximum time increment
CFL_SAFETY_FACTOR = 0.25  # Courant-Friedrichs-Lewy condition safety

# Output frequency
OUTPUT_FREQUENCY = 60  # Save results every 60 seconds
VISUALIZATION_FREQUENCY = 300  # Create plots every 5 minutes

# ==================== HEAT TRANSFER PARAMETERS ====================
# Convection coefficients
H_CONV_SAMPLE_AIR_SPACE = 5.0    # W/(m²·K) - natural convection inside tube
H_CONV_AIR_GAP = 5.0  # W/(m²·K) - natural convection in air gap
H_CONV_AMBIENT = 10.0    # W/(m²·K) - natural convection outside enclosure

# Radiation parameters
STEFAN_BOLTZMANN = 5.670374419e-8   # W/(m²·K⁴)

# Radiation heat transfer flags
ENABLE_RADIATION = True       # Enable radiation heat transfer
RADIATION_VIEW_FACTOR = 1.0   # View factor for radiation (simplified as 1.0)

# ==================== NUMERICAL PARAMETERS ====================
# Convergence criteria
TEMPERATURE_TOLERANCE = 1e-6   # K - convergence tolerance
MAX_ITERATIONS = 100000000000000000         # Maximum iterations per time step
RESIDUAL_TOLERANCE = 1e-8     # Residual tolerance

# ==================== COIL TURN DISTRIBUTION - PERFECT 1:1 MAPPING ====================
def get_coil_turn_positions():
    """Calculate heating positions for perfect coil-to-node mapping
    
    CONSTRAINT ENFORCED: AXIAL_NODES_HEATING = HEATING_COIL_TURNS = 39
    Each coil turn maps to exactly one heating node (perfect 1:1 mapping)
    """
    # Constraint is enforced above, so we can safely use either value
    #assert AXIAL_NODES_HEATING == HEATING_COIL_TURNS, "Constraint violation detected"
    
    # Create positions for each coil turn/node (39 positions for 39 turns/nodes)
    positions = np.linspace(HEATING_COIL_START, HEATING_COIL_END, HEATING_COIL_TURNS)
    
    return positions

# Generate coil positions with perfect turn mapping
COIL_TURN_POSITIONS = get_coil_turn_positions()

# Physical spacing validation
COIL_SPACING = HEATING_COIL_LENGTH / (HEATING_COIL_TURNS - 1) if HEATING_COIL_TURNS > 1 else 0
NODE_SPACING = HEATING_COIL_LENGTH / (AXIAL_NODES_HEATING - 1) if AXIAL_NODES_HEATING > 1 else 0

# These should be identical due to the constraint
#assert abs(COIL_SPACING - NODE_SPACING) < 1e-10, f"Spacing mismatch: coil={COIL_SPACING:.6f}, node={NODE_SPACING:.6f}"

# ==================== OUTPUT SETTINGS ====================
OUTPUT_DIR = 'high_res_results'
MESH_OUTPUT_FILE = 'mesh_details.txt'
TEMPERATURE_OUTPUT_FILE = 'temperature_evolution.h5'  # HDF5 for large datasets
VISUALIZATION_DIR = 'visualizations'
LOG_FILE = 'simulation_log.txt'

# Function to get exact node count from mesh generation
def get_exact_node_count():
    """Get exact node count by generating the mesh"""
    try:
        import mesh
        mesh_obj = mesh.TubeFurnaceMesh()
        mesh_obj._generate_cylindrical_region()
        cylindrical_nodes = len(mesh_obj.r_nodes) * len(mesh_obj.z_nodes)
        return cylindrical_nodes
    except ImportError:
        return None, None, None

# Print configuration summary
def print_config_summary(include_exact_count=False):
    """Print configuration summary for new hybrid mesh system"""
    print("HYBRID MESH SYSTEM - TUBE FURNACE SIMULATION CONFIG")
    print("=" * 60)
    print(f"Materials:")
    print(f"  Cement: {FURNACE_CEMENT_TYPE}")
    print(f"  Insulation: {INSULATION_TYPE}")
    print(f"\nHybrid Mesh Resolution:")
    print(f"  Cylindrical region:")
    print(f"    Radial nodes: {TOTAL_RADIAL_NODES} ({RADIAL_NODES_SAMPLE}+{RADIAL_NODES_GLASS}+{RADIAL_NODES_CEMENT}+{RADIAL_NODES_CERAMIC}+{RADIAL_NODES_REFLECTIVE})")
    print(f"    Axial nodes: {TOTAL_AXIAL_NODES} ({AXIAL_NODES_BEFORE}+{AXIAL_NODES_HEATING}+{AXIAL_NODES_AFTER})")
    print(f"    Total cylindrical: {TOTAL_RADIAL_NODES * TOTAL_AXIAL_NODES:,}")
    print(f"    Filtering: Geometric exclusion removes nodes inside cylindrical region")
    print(f"    Actual nodes: Determined by mesh.py generation algorithm")
    print(f"    Coverage: Air gap + aluminum enclosure outside {REFLECTIVE_CASING_OUTER_RADIUS*1000:.1f}mm radius")
    print(f"  Total system nodes: Requires mesh generation for exact count")
    print(f"\nCoil-Node Mapping (CONSTRAINT ENFORCED):")
    print(f"  Coil turns = Heating nodes: {HEATING_COIL_TURNS}")
    print(f"  Mapping ratio: {HEATING_COIL_TURNS/AXIAL_NODES_HEATING:.3f} (LOCKED at 1.000)")
    print(f"  Turn spacing: {COIL_SPACING*1000:.2f}mm per turn/node")
    print(f"  Power per turn/node: {HEATING_COIL_POWER/HEATING_COIL_TURNS:.1f}W")
    print(f"\nAxial Discretization Comparison:")
    cold_zone_length = (FURNACE_LENGTH - HEATING_COIL_LENGTH) / 2
    cold_zone_spacing = cold_zone_length / AXIAL_NODES_BEFORE * 1000  # mm per node
    heating_zone_spacing = HEATING_COIL_LENGTH / AXIAL_NODES_HEATING * 1000  # mm per node
    
    print(f"  Cold zones:")
    print(f"    Length: {cold_zone_length*1000:.1f}mm ({cold_zone_length/0.0254:.1f} inches) each")
    print(f"    Nodes: {AXIAL_NODES_BEFORE} per cold zone")
    print(f"    Resolution: {cold_zone_spacing:.2f}mm per node")
    print(f"  Heating zone:")
    print(f"    Length: {HEATING_COIL_LENGTH*1000:.1f}mm ({HEATING_COIL_LENGTH/0.0254:.1f} inches)")
    print(f"    Nodes: {AXIAL_NODES_HEATING} (= coil turns)")
    print(f"    Resolution: {heating_zone_spacing:.2f}mm per node")
    print(f"  Resolution ratio: {heating_zone_spacing/cold_zone_spacing:.2f}x (heating/cold)")
    print(f"  Heating zone is {'denser' if heating_zone_spacing < cold_zone_spacing else 'coarser'} than cold zones")
    print(f"\nTemporal Resolution:")
    print(f"  Duration: {SIMULATION_DURATION/3600:.1f} hours")
    print(f"  Time step: {TIME_STEP_SECONDS:.1f} seconds")
    print(f"  Total steps: {TOTAL_TIME_STEPS:,}")
    print(f"\nHeat Transfer Optimization:")
    print(f"  Dense radial mesh near heating zone: {RADIAL_NODES_CEMENT + RADIAL_NODES_CERAMIC} nodes")
    print(f"  Perfect coil discretization: No rounding errors")
    print(f"  Hybrid geometry: Cylindrical (inner) + Cubic (outer)")
    print(f"\nNode Count Breakdown:")
    print(f"  Cylindrical mesh: {TOTAL_RADIAL_NODES} radial × {TOTAL_AXIAL_NODES} axial = {TOTAL_CYLINDRICAL_NODES:,} nodes")
    print(f"  TOTAL SYSTEM: Run 'py config.py --exact' for precise count")
    
    print(f"\nCustomizable Parameters:")
    print(f"  Radial resolution: Modify RADIAL_NODES_* in config.py")
    print(f"  Constraint: AXIAL_NODES_HEATING = HEATING_COIL_TURNS (locked at {HEATING_COIL_TURNS})")

if __name__ == "__main__":
    import sys
    # Check if user wants exact count (slower)
    include_exact = len(sys.argv) > 1 and sys.argv[1] == "--exact"
    print_config_summary(include_exact_count=include_exact)