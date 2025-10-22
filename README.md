# Tube Furnace Heat Simulation(Functional)

Currently Working File:
config.py
hyperbolic.py
material.py
mesh.py
requirements.txt
solver.py

# Pseudo-3D/2D Cylindrical Finite Volume Method (FVM) – Lumped Element Model Hybrid Heat Simulater

A customized heat simulation model developed for small-scale tube furnaces, where traditional commercial thermal analysis tools are inadequate due to severe time step constraints imposed by the Courant–Friedrichs–Lewy (CFL) condition.

This hybrid model combines a pseudo-3D/2D cylindrical Finite Volume Method (FVM) with a Lumped Element approach, optimizing both computational efficiency and physical accuracy.

## Current System: **Pseudo-3D/2D Cylindrical 5-Layer Tube Furnace System Design**

This simulation models a **5-layer concentric tube furnace system** with precise heat transfer physics:

### **Layer Structure (from center outward):**

```
Layer 1: AIR (Sample Space)          │ r = 0 → 10.9982mm                     (10.9982mm)
Layer 2: QUARTZ GLASS                │ r = 10.9982mm → 12.7mm                (1.7018mm)
Layer 3: KANTHAL COIL (embedded)     │ r = 12.7mm → 12.955mm  (heating zone) (0.2550mm)
Layer 4: IMPERIAL CEMENT             │ r = 12.955mm → 15.7mm                 (2.7450mm)
Layer 5: LYRUFEXON CERAMIC FIBER     │ r = 15.7mm → 117.5mm                  (101.8mm)
Layer 6: REFLECTIVE ALUMINUM CASING  │ r = 117.5mm → 118.5mm                 (1mm)
Layer 7: SEALED AIR GAP              │ r = 118.5mm → 148.5mm                 (30mm)
Layer 7: ALUMINUM 5052-H32           │ r = 148.5mm → 151.5mm                 (3mm)
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
- **Radial**: Through glass (k=1.4 W/m·K), cement (k=0.6-1.85 W/m·K), ceramic fiber (k=0.045-0.42 W/m·K)
- **Axial**: Along glass tube length, distributing heat from coil zone to cold zones
- **Temperature Dependence**: All thermal conductivities vary with temperature

**B. Convection:**
- **Inner tube**: Natural convection (h=5 W/m²·K) - sample space to glass
- **Air gap**: Natural convection (h=5 W/m²·K) - thermal isolation
- **Outer surface**: Natural convection (h=10 W/m²·K) - enclosure to ambient

**C. Radiation:**
- **Glass surface**: High emissivity (ε=0.88-0.94) - significant radiation at high temperatures
- **Reflective casing**: Low emissivity (ε=0.02-0.05) - **radiation barrier** (95% reflection)
- **Aluminum enclosure**: Low emissivity (ε=0.05-0.09) - minimal heat loss

#### **4. Heat Losses:**

**Primary Heat Loss Paths:**
1. **Radial losses**: Through insulation → air gap → enclosure wall → ambient
2. **Axial losses**: Through cold zones at tube ends
3. **Radiation losses**: Surface radiation (minimized by reflective casing)

**Heat Loss Reduction Strategies:**
- **Ceramic fiber insulation**: 102.3mm thick, very low conductivity (k=0.045-0.22 W/m·K)
- **Reflective barrier**: 95% radiation reflection, prevents radiative losses
- **Air gap**: 30mm thermal break, prevents direct leak to ambient atmosphere

### **Boundary Conditions:**

#### **Inner Boundary (r=0, centerline):**
```
∂T/∂r = 0  (symmetry condition)
Convection to Inner Glass Tube Space
Q = h·A·(T_glass - T_sample)
Lump node temperature change regards to heat flux
dT/dt​ = Q/mCv​

```
#### **Internal Region (r=0, multi-layer cylinderical region):**
```
Conduction across internal cell
∂T/∂t = 1/ρCv·(​​∂/∂r​(k(T)·∂T∂r​))
where k(T) at interface is treated with weighted average from interpolated k
k(T) = 1/((k1/Δr1)+(k2/Δr2))⁻¹
```
#### **Heating Zone (Coil positions):**
```
480W distributed over 39 positions
Q = 480W/V_smeared 
where
V_smear = π·r2²·r1²·z
dT/dt​ = V_cell·Q/mCv​
```
#### **Intermediate Boundary (Reflective surface):**
```
Combined convection + radiation across layer (T unknown°C)
Q = h·A·(T_surf - T_air) + ε·σ·A·(T_surf⁴ - T_casing⁴)
Lump node temperature change regards to heat flux
dT/dt​ = Q/mCv​
```
#### **Outer Boundary (enclosure surface):**
```
Combined convection + radiation to ambient (25°C)
Q = h·A·(T_air - T_ambient) + ε·σ·A·(T_casing⁴ - T_ambient⁴)
Lump node temperature change regards to heat flux
dT/dt​ = Q/mCv​
```
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

### **Key Assumptions:**

# Quasi-Steady-State Radiation

- Radiation heat transfer is modeled using a quasi-steady-state grey body approximation, with constant surface emissivity assumed over time.
- Radiation exchange is considered to reach steady conditions within each time step, and is decoupled from transient conduction behavior.
- Only surface-to-surface radiation is modeled; no absorption, scattering, or transmittance occurs through material layers, which are treated as perfectly opaque.
- The air medium is non-participating in radiation; it does not absorb, emit, or scatter thermal radiation.
- Back radiation from colder surfaces is neglected, and view factor geometry is not considered, reducing the radiation model to a one-way, unidirectional heat transfer mechanism.
- No radiation exchange is considered across the multi-layer cylindrical region; thermal conduction is the dominant heat transfer mechanism within the solid structure.

# Perfect Thermal Contact

- No thermal contact resistance is considered at interfaces between adjacent material layers.
- Interfaces are assumed to be in perfect thermal contact, allowing continuous and uninterrupted heat conduction across boundaries.
- Thermal properties are consistent across interfaces, with no temperature drop or discontinuity in heat flux between layers.

# Uniform Coil Heating

- Consistent volumetric heat generation is applied uniformly within the designated heating zone.
- Resistive heating elements (e.g., heating coils) are assumed to produce a uniform radial power distribution along their axial length.
- The helical geometry of coils and any localized heating effects (e.g., hotspots or concentrated nodes) are neglected, simplifying the model to a spatially uniform heat source.
- Implementation of Gaussian-distributed heating profiles is reserved for future model versions to capture localized power variations more accurately.

# Simplified Thermal Environment and Convection Modeling

- Convective heat loss is modeled using Newton’s law of cooling, with no external fans or forced airflow; only natural convection is considered.
- The air domain is treated as a uniform lumped node, which reaches thermal equilibrium quasi-instantaneously within each time step (quasi-steady-state assumption).
- A fixed, user-defined convective heat-transfer coefficient is used; no empirical correlations or dynamic calculation of 
h is included.
- No computational fluid dynamics (CFD) or air movement modeling is performed; airflow and buoyancy effects are not resolved.
- The thermal response of the air is based on a constant-volume, ideal gas approximation, with thermal diffusivity interpolated at each temperature step.
- No enthalpy changes (e.g., from phase change or air moisture) are included in the air domain; only sensible heat (via temperature) is considered.
- No mass flow or heat loss is included within the sample air space, but the framework allows for future integration of pressurized mass flow or open-system behavior.

# Material Homogeneity Within Layers

- Each material layer is assumed to be homogeneous and isotropic, with uniform thermal properties throughout its volume.
- Spatial variations in material composition, porosity, or fiber alignment within a single layer are not considered.
- Localized thermal anomalies (e.g., hot spots, thermal bridges, or short circuits) caused by microstructural inhomogeneities are neglected.

# No Chemical or Phase Reactions

- The model simulates pure heat transfer without coupling to any chemical reactions or material transformations.
- Effects from air chemistry, oxidation, combustion, pyrolysis, decomposition, or thermal degradation are excluded.
- Phase changes (e.g., melting, vaporization, condensation) and associated latent heat effects are not modeled.
- All materials are assumed to remain in their initial solid phase throughout the simulation.

# Adiabatic Axial Boundary (z-axis)

- The furnace is modeled as a segment of an infinitely long cylinder wrapped with finite dimension of heating element. This setting trace heat propagation in the cold zone and simplifies the domain by neglecting end effects and axial heat loss.

- No heat flux is permitted through the axial boundaries. This assumption neglects end effects and simplifies the domain to focus on radial and circumferential heat transfer.

- However, the model allows for some degree of axial temperature profiling within the active heating region, enabling more accurate representation of local gradients while maintaining adiabatic conditions at the boundaries.
---

##  **Installation & Setup**

### ** Manual Setup**

#### **Step 1: Clone Repository**
```powershell
git clone https://github.com/llhtimlam/Hacker-Fab---Tube-Furnace-Heat-Simulationn.git
cd "Hacker Fab - Tube Furnace Heat Simulation"
```

#### **Step 2: Environment & Dependencies**
```powershell or terminal inside your python IDE
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install all dependencies (optimized versions to avoid conflicts)
pip install -r requirements.txt

# Verify installation
python -c "import numpy, scipy, matplotlib, plotly, h5py; print('✅ All packages ready')"
```
#### **Step 3: Run Simulation**
solver.py

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
├── 📈 hyperbolic.py               # Optimize Mesh Spacing
├── 🧱 materials.py                # 7-layer material properties database
├── 🕸️ mesh.py                     # High-resolution mesh generation
├── 🧮 solver.py                   # Heat transfer physics solver
├── 🚀 COMPLETE_SETUP.bat          # One-click installation & launcher (WiP)
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

Support interpolation for temperature dependent properties

### **Mesh Geometry:**

Review `mesh.py` for mesh spatial geometry

Edit `hyperbolic.py` for optimizing number of node with hyperbolic tangent spacing (S curve, fine in edge and coarse in middle)

Run `hyperbolic.py`, if detailed result is needed, toggle Debug Mode in Config.py for exporting the Hyperbolic Spacing Grid

Located in Debug folder

##  **Simulation Workflow Overview**

1. Edit config.py, materials.py for desired setting

Make sure Initial Temperature Kanthal > Cylindrical Region > Lump Node to avoid starting point numerical instability

2. Toggle Import/Export Mode in Config.py

If Import Mode = False in config.py
Mesh will be generated based on config.py setting and skip to next step

If Import Mode = True in config.py
Ensure the following files are placed in Import folder and make sure config.py and materials.py is identical to previous run:
r_centers.csv    r_faces.csv
z_centers.csv    z_faces.csv
Cylindrical Region Final Step Temperature.csv
material_map_cylindrical_centered.csv

3. Run `solver.py`

If Import Mode = True:
Type based on the prompted in the terminal for Lump Node Region Temperature Setup
where it can be retrieved in any column in the Tplot.csv:
Sample Air Space Temperature    1st row       (index 0)
Air Gap Temperature             last 2nd row  (index -1)
Aluminium Casing Temperature    last row      (index -2)

4. Output Results

Main Output:
Located in Solver Graphics and Data folder
h5.file contains all time data and png.file transient profile at last iteration

Optional Output:
Export folder for future import as a checkpoint (Backup config.py and materials.py)
Debug folder for tracing heat flux and detailed thermal data

**Last Updated**: October 2025  
**Project Status**:  **Functional, WIP for extra feature**
**Dependencies**:  **Conflict-Free** - Tested package versions

### **Future Implementation

Research Implicit Method
Implement Steady State Analysis
Implement mass flow simulation in sample air space
Modify radiation model
Fix graphic for better illustration of node layout
Improve modularity of each file
Revamp the code in C environment