# Tube Furnace Heat Simulation(WIP)

Currently Working File:
config.py
material.py
mesh.py
requirements.txt
solver.py (graphic function is not fully implemented)

# Pseudo-3D/2D Cylindrical Finite Volume Method (FVM) – Lumped Element Model Hybrid Heat Simulater

A customized heat simulation model developed for small-scale tube furnaces, where traditional commercial thermal analysis tools are inadequate due to severe time step constraints imposed by the Courant–Friedrichs–Lewy (CFL) condition.

This hybrid model combines a pseudo-3D/2D cylindrical Finite Volume Method (FVM) with a Lumped Element approach, optimizing both computational efficiency and physical accuracy.

# Key Features

# Comprehensive Heat Transfer Modeling

- Models conduction, convection, and radiation heat transfer mechanisms.

Supports various boundary conditions:

- Dirichlet (Fixed Temperature)
- Neumann (Fixed Heat Flux)
- Robin (Convective/Radiative Heat Transfer)

- Surface emissivity, convective coefficients, and ambient conditions are fully configurable.

# Hybrid Numerical Modeling

- Spatially resolved finite volume discretization in cylindrical and annular domains.

- Integrated with lumped-element thermal modeling for components where full spatial resolution is unnecessary.

- Supports axisymmetric configurations with scalable radial and axial resolution.

# CFL Condition Bypass

- Explicit handling of time step constraints to avoid CFL limitations.

- Allows relatively large, and practical time steps while maintaining numerical stability.

- Adjustable mesh density for tailored spatial resolution and computational cost.

# Flexible Geometry Configuration

- Supports solid and multi-layered hollow cylindrical geometries.

- Fully customizable dimensions, mesh granularity, and layer configurations.

- Material properties can be defined as constants or functions of temperature (with interpolation support).

- Configurable internal heat generation for simulating resistive heating, chemical reactions, or other volumetric sources.

# Target Applications

- Simulation of thermal profiles in custom or small-scale tube furnaces.

- Prototyping and virtual testing of heating profiles in laboratory-scale reactors.

- Modeling of thermal response in high-temperature material processing environments.

## Current System: **Pseudo-3D/2D Cylindrical 5-Layer Tube Furnace System Design**

This simulation models a **5-layer concentric tube furnace system** with precise heat transfer physics:

### **Layer Structure (from center outward):**

```
Layer 1: AIR (Sample Space)          │ r = 0 → mm
Layer 2: QUARTZ GLASS                │ r = mm → mm
Layer 3: ANTHAL COIL (embedded)      │ r = mm → mm  (heating zone)
Layer 4: IMPERIAL CEMENT             │ r = mm → mm
Layer 5: LYRUFEXON CERAMIC FIBER     │ r = mm → mm
Layer 6: REFLECTIVE ALUMINUM CASING  │ r = mm → mm
Layer 7: SEALED AIR GAP              │ r = mm → mm
Layer 7: ALUMINUM 5052               │ r = mm → mm
```

### **Heat Transfer Flow Analysis:**

#### **1. Heat Generation (Layer 4 - Kanthal Coil):**
- **Power Input**: 480W electrical power
- **Heating Zone**: Z = 25.4mm to 279.4mm (254mm span)
- **Distribution**: 39 physical turns → 39 computational positions (continuous coverage)
- **Temperature**: 
- **Heat Mechanism**: Joule heating (I²R losses) converted to thermal energy

#### **2. Heat Conduction Path:**

**Radial Heat Flow:**
```
Coil (480W) → Cement → Glass → Air (sample)
            ↓
    Ceramic Fiber (insulation)
            ↓
    Reflective Casing (radiation barrier)
            ↓
    Air Gap (thermal break)
            ↓
    Aluminum Enclosure → Ambient
```

**Axial Heat Flow:**
```
Cold Zone (Z=0-25mm) ← Heat Conduction → Heating Zone (Z=25-279mm) ← Heat Conduction → Cold Zone (Z=279-304mm)
```

#### **3. Heat Transfer Mechanisms:**

**A. Conduction (Primary):**
- **Radial**: Through glass (k=1.2-2.0 W/m·K), cement (k=0.6-1.85 W/m·K), ceramic fiber (k=0.045-0.22 W/m·K)
- **Axial**: Along glass tube length, distributing heat from coil zone to cold zones
- **Temperature Dependence**: All thermal conductivities vary with temperature

**B. Convection:**
- **Inner tube**: Natural convection (h=10 W/m²·K) - sample space to glass
- **Air gap**: Natural convection (h=15 W/m²·K) - thermal isolation
- **Outer surface**: Natural convection (h=25 W/m²·K) - enclosure to ambient

**C. Radiation:**
- **Glass surface**: High emissivity (ε=0.9) - significant radiation at high temperatures
- **Ceramic fiber**: High emissivity (ε=0.8) - radiative heat transfer
- **Reflective casing**: Low emissivity (ε=0.05) - **radiation barrier** (95% reflection)
- **Aluminum enclosure**: Low emissivity (ε=0.05-0.09) - minimal heat loss

#### **4. Heat Losses:**

**Primary Heat Loss Paths:**
1. **Radial losses**: Through insulation → air gap → enclosure wall → ambient
2. **Axial losses**: Through cold zones at tube ends
3. **Radiation losses**: Surface radiation (minimized by reflective casing)

**Heat Loss Reduction Strategies:**
- **Ceramic fiber insulation**: 102.3mm thick, very low conductivity (k=0.045-0.22 W/m·K)
- **Reflective barrier**: 95% radiation reflection, prevents radiative losses
- **Air gap**: 30mm thermal break, prevents conductive heat bridge
- **Continuous heating**: 41 positions eliminate cold spots in heating zone

### **Boundary Conditions:**

#### **Inner Boundary (r=0, centerline):**
```
∂T/∂r = 0  (symmetry condition)
-k(∂T/∂r) = h_conv(T_glass - T_sample)
Convection to Inner Glass Tube Space
```
#### **Intermediate Boundary (Reflective surface):**
```
-k(∂T/∂r) = h_conv(T_surface - T_unknown) + ε·σ(T_surface⁴ - T_unknown⁴)
Combined convection + radiation across layer (T unknown°C)
```
#### **Outer Boundary (enclosure surface):**
```
-k(∂T/∂r) = h_conv(T_unknown - T_ambient) + ε·σ(T_unknown⁴ - T_ambient⁴)
Combined convection + radiation to ambient (25°C)
```

#### **Axial Boundaries (tube ends):**
```
Z=0 and Z=304.8mm: Combined convection + radiation to ambient
```

#### **Heating Zone (Coil positions):**
```
Q_coil = 480W distributed over 39 positions
Power density varies with coil turn spacing
```

### **Key Assumptions:**

# Quasi-Steady-State Radiation
Radiation heat transfer is modeled using a quasi-steady-state grey body approximation, surface emissivity remains constant and radiation exchange reaches steady conditions within each time step.
Treating radiation only in one direction.
No absorption and transmitance across layer.

# Perfect Thermal Contact
No thermal contact resistance is considered between adjacent material layers, interfaces are assumed to be in perfect thermal contact, allowing uninterrupted heat conduction with consistent thermal properties.

# Uniform Coil Heating
Resistive heating elements (e.g., heating coils) are assumed to have uniform power distribution along their length, resulting in consistent volumetric heat generation within the heating zone.

# Natural Convection Only
Heat transfer to the environment via convection is assumed to occur under natural convection conditions; no forced airflow or external cooling is applied. No fluid mechanic is considered.

# Material Homogeneity Within Layers
Each material layer is assumed to be homogeneous, with isotropic thermal properties. Spatial variation within a single layer is not considered.

# No Chemical or Phase Reactions
The model simulates pure heat transfer processes. Effects from chemical reactions (e.g., combustion, oxidation, decomposition) or phase changes (e.g., melting, evaporation) are excluded.

# Adiabatic Axial Boundary (z-axis)
The furnace is modeled as a segment of an infinitely long cylinder by applying adiabatic boundary conditions at the ends along the z-axis. This setting trace heat propagation in the cold zone and simplifies the domain by neglecting end effects and axial heat loss.

---

##  **Installation & Setup**

### ** Manual Setup**

#### **Step 1: Clone Repository**
```powershell
git clone https://github.com/llhtimlam/Hacker-Fab---Tube-Furnace-Heat-Simulationn.git
cd "Hacker Fab - Tube Furnace Heat Simulation"
```

#### **Step 2: Environment & Dependencies**
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install all dependencies (optimized versions to avoid conflicts)
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, matplotlib, plotly, h5py; print('✅ All packages ready')"
```
#### **Step 3: Run Simulation**
*Type py solver.py in Terminal**

**Package Versions (Conflict-Free):**
```
numpy>=1.21.0,<2.0.0
scipy>=1.7.0,<1.12.0
matplotlib>=3.5.0,<4.0.0
h5py>=3.7.0,<4.0.0
tqdm>=4.62.0,<5.0.0
pandas>=1.3.0,<3.0.0
plotly>=5.0.0
```

### ** Essential Files (Everything You Need):**

```
Tube Furnace Heat Simulation/
├── 
├── ⚙️ config.py                   # System configuration parameters
├── 🧱 materials.py                # 7-layer material properties database
├── 🕸️ mesh.py                     # High-resolution mesh generation
├── 🧮 solver.py                   # Heat transfer physics solver
├── 🚀 COMPLETE_SETUP.bat          # One-click installation & launcher
├── 📋 requirements.txt            # Optimized dependencies (conflict-free)
├── 📖 README.md                   # This documentation
├── 📁 .venv/                      # Python virtual environment
├── 📁 high_resolution_backup/     # Development backup
```

### ** Optimized Dependencies (`requirements.txt`)**
**Conflict-free package versions:**
```
numpy>=1.21.0      # Core numerical computing
scipy>=1.7.0       # Scientific computing
matplotlib>=3.5.0  # Static plotting  
h5py>=3.7.0        # Data storage
tqdm>=4.62.0       # Progress bars
pandas>=1.3.0      # Data handling
plotly>=5.0.0      # Interactive visualizations
```

**Optional performance packages** (commented out by default):
```
# numba>=0.56.0    # JIT compilation for speed
# cupy>=10.0.0     # GPU acceleration
```

---

##  **Configuration Guide**

### **Modifying System Parameters:**

Edit `config.py` to customize the simulation setup:

### **Material Properties:**

Edit `materials.py` for editing material properties

All materials in `materials.py` must have temperature-dependent properties:
- Thermal conductivity k(T)
- Specific heat cp(T) 
- Density ρ(T)

Support interpolation with cubic interpolating that can be customized

### **Mesh Geometry:**

Edit `mesh.py` for editing mesh spatial geometry

##  **Simulation Output**

### ** Main Output: h5.file and np.plot diagram**

Run `solver.py` 

**Last Updated**: October 2025  
**Project Status**:  **WIP**
**Dependencies**:  **Conflict-Free** - Tested package versions