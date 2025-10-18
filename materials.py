"""
High-Resolution Material Properties Database
Temperature-dependent properties for all materials with interpolation
"""

import numpy as np
from scipy.interpolate import interp1d
import json
# import h5py  # Optional - only needed for database saving

class TubeFurnaceMaterials:
    """Advanced material properties with high-resolution temperature dependence"""
    
    def __init__(self):
        self.materials = self._initialize_materials()
        self.material_index_to_name = {
        0: 'air',
        1: 'quartz_glass',
        2: 'kanthal_coil',
        3: 'imperial_cement',
        4: 'lyrufexon_ceramic',
        5: 'reflective_aluminum',
        6: 'air',
        7: 'aluminum_5052'
        }
        self.interpolators = self._create_interpolators()
        self.get_thermal_properties_vec = np.vectorize(self.get_thermal_properties_for_vectorize, otypes=[float, float, float])
        self.get_emissivity_vec = np.vectorize(self.get_emissivity_for_vectorize, otypes=[float])
        
    def _initialize_materials(self):
        """Initialize comprehensive material database"""
        materials = {}
        
        # ==================== AIR (Sample space, air gaps) ====================
        materials['air'] = {
            'density': 1.225,  # kg/m³ at STP
            'temp_range': np.array([250, 300, 400, 500, 600, 800, 1000, 1200, 1500]),
            'thermal_conductivity': np.array([0.0209, 0.0263, 0.0338, 0.0407, 0.0469, 0.0573, 0.0667, 0.0754, 0.0883]),
            'specific_heat': np.array([1006, 1007, 1014, 1030, 1051, 1099, 1142, 1179, 1234]), #Constant Pressure for Air Flow Systems (Need mass flow temperature in out), for current model, constant volume is considered
            'dynamic_viscosity': np.array([1.488e-5, 1.846e-5, 2.286e-5, 2.671e-5, 3.018e-5, 3.625e-5, 4.152e-5, 4.621e-5, 5.378e-5])
        }

        # ==================== QUARTZ GLASS ====================
        materials['quartz_glass'] = {
            'density': 2214.39238,  # kg/m³
            'temp_range': np.array([293.15,313.15,373.15, 473.15]),
            'thermal_conductivity': np.array([1.4, 1.4, 1.4, 1.4]),  # Wm-1K-1
            'specific_heat': np.array([749, 749, 749, 749]),  # Jkg-1K-1
            'emissivity': 0.93,
            'max_temperature': 1366.4833  # 2000°F
        }
        
        # ==================== IMPERIAL HIGH TEMPERATURE CEMENT ====================
        materials['imperial_cement'] = {
            'density': 1860,  # kg/m³
            'temp_range': np.array([293.15,313.15, 373.1, 473.15]),
            'thermal_conductivity': np.array([0.72,0.72,0.72,0.72]),  # Wm-1K-1
            'specific_heat': np.array([800,800,800,800]),  # Jkg-1K-1
            'emissivity': 0.96,
            'max_temperature': 1755.15  # 1482°C
        }
        
        # ==================== LYRUFEXON CERAMIC FIBER INSULATION ====================
        materials['lyrufexon_ceramic'] = {
            'density': 96,  # kg/m³ (low density blanket) 128 kgm-3 high density blanket dry density 190-290 kgm-3
            'temp_range': np.array([873.15, 1073.15, 1273.15, 1473.15]),
            'thermal_conductivity': np.array([0.14, 0.20, 0.29, 0.42]), #128kgm-3 0.13 0.18 0.26 0.36 Wm-1K-1
            'specific_heat': np.array([1246, 1246, 1246, 1246]),  # Jkg-1K-1
            'emissivity': 0.85, # Eyeball estimate for ceramic wool https://metalurji.mu.edu.tr/Icerik/metalurji.mu.edu.tr/Sayfa/MME%202506-Refractory%20Materials-8.pdf
            'max_temperature': 1873.15,  # 1600°C
        }
        
        # ==================== KANTHAL HEATING COIL ====================
        materials['kanthal_coil'] = {
            'density': 7100,  # kg/m³
            'temp_range': np.array([293.15, 323.15, 473.15, 673.15, 873.15, 1073.15, 1273.15, 1473.15, 1673.15]),
            'thermal_conductivity': np.array([9.430309439442341, 11, 16.433517281440615, 19.24223061283765, 20, 22, 26, 27, 35]),
            'specific_heat': np.array([460, 486.2886729991089, 560, 630, 750, 710, 720, 740, 800]),
            'emissivity': 0.70, # Emissivity - fully oxidized material
            'max_temperature': 1673  # 1400°C = 1673K continuous operation
        }
        
        # ==================== REFLECTIVE ALUMINUM CASING ====================
        materials['reflective_aluminum'] = {
            'density': 2725,  # kg/m³ - Pure aluminum sheet
            'temp_range': np.array([298.15,313.15,373.15,473.15]),
            'thermal_conductivity': np.array([235,235,235,235]),
            'specific_heat': np.array([880,880,880,880]),
            'emissivity': 0.03,  # 0.03-0.06 Very low emissivity for polished reflective surface
            'reflectivity': 0.94,  # High reflectivity (94-97%)
            'surface_finish': 'polished',
        }
        
        # ==================== ALUMINUM 5052-H32 CUBIC ENCLOSURE ====================
        materials['aluminum_5052'] = {
            'density': 2680,  # kg/m³
            'temp_range': np.array([298.15,313.15,373.15,473.15]),
            'thermal_conductivity': np.array([138,138,138,138]),
            'specific_heat': np.array([880,880,880,880]),
            'emissivity': 0.09,  # Natural finish
            'emissivity_anodized': 0.85   # Black anodized finish
        }
        
        return materials

    def _create_interpolators(self):
        """Create interpolation functions for all material properties"""
        interpolators = {}
        
        for material_name, properties in self.materials.items():
            interpolators[material_name] = {}
            temp_range = properties['temp_range']
            
            # Create interpolators for each property
            for prop_name, values in properties.items():
                if prop_name != 'temp_range' and isinstance(values, np.ndarray):
                    # Use cubic interpolation for smooth derivatives
                    interpolators[material_name][prop_name] = interp1d(
                        temp_range, values, 
                        kind='cubic', 
                        bounds_error=False, 
                        fill_value='extrapolate'
                    )
        
        return interpolators
    
    def get_property(self, material, property_name, temperature):
        """Get material property at specific temperature with interpolation"""
        # Clamp temperature to reasonable bounds
        temp_clamped = np.clip(temperature, 250, 2000)
        
        if material in self.interpolators:
            if property_name in self.interpolators[material]:
                return float(self.interpolators[material][property_name](temp_clamped))
            elif property_name in self.materials[material]:
                # Return constant value if not temperature dependent
                return self.materials[material][property_name]
        
        raise ValueError(f"Property '{property_name}' not found for material '{material}'")
    
    @staticmethod
    def get_thermal_properties_for_vectorize(materials_obj, material_index, temperature):
        """
        Static helper for vectorization. Retrieves properties for a single material index and temperature.
        """
        material_name = materials_obj.material_index_to_name.get(material_index)
        if material_name is None:
            raise ValueError(f"Invalid material index: {material_index}")
        
        return materials_obj.get_thermal_properties(material_name, temperature)

    def get_thermal_properties(self, material, temperature):
        """Get thermal conductivity, density, and specific heat"""
        k = self.get_property(material, 'thermal_conductivity', temperature)
        rho = self.get_property(material, 'density', temperature)
        cp = self.get_property(material, 'specific_heat', temperature)
        return k, rho, cp
    
    def get_thermal_diffusivity(self, material, temperature):
        """Calculate thermal diffusivity α = k/(ρ·cp)"""
        k, rho, cp = self.get_thermal_properties(material, temperature)
        return k / (rho * cp)
    
    @staticmethod
    def get_emissivity_for_vectorize(materials_obj, material_index):
        """
        Static helper for vectorization. Retrieves properties for a single material index and temperature.
        """
        material_name = materials_obj.material_index_to_name.get(material_index)
        if material_name is None:
            raise ValueError(f"Invalid material index: {material_index}")
        
        return materials_obj.get_emissivity(material_name)
    
    def get_emissivity(self, material):
        """Get emissivity for a material"""
        if material in self.materials:
            return self.materials[material].get('emissivity')
        raise ValueError(f"Property 'emissivity' not found for material '{material}'")

    def get_all_materials(self):
        """Return list of all available materials"""
        return list(self.materials.keys())
    
    def get_temperature_range(self, material):
        """Get valid temperature range for material"""
        if material in self.materials:
            temp_range = self.materials[material]['temp_range']
            return temp_range.min(), temp_range.max()
        return None, None
    
    def save_database(self, filename):
        """Save material database to JSON file for portability"""
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for material, properties in self.materials.items():
            json_data[material] = {}
            for prop_name, values in properties.items():
                if isinstance(values, np.ndarray):
                    json_data[material][prop_name] = values.tolist()
                else:
                    json_data[material][prop_name] = values
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def validate_temperature_range(self, material, temperature):
        """Check if temperature is within valid range for material"""
        min_temp, max_temp = self.get_temperature_range(material)
        if min_temp is not None and max_temp is not None:
            if temperature < min_temp or temperature > max_temp:
                print(f"Warning: Temperature {temperature:.1f}K outside range [{min_temp:.1f}, {max_temp:.1f}]K for {material}")
                return False
        return True
    
    def get_material_summary(self):
        """Print summary of all materials and their properties"""
        print("HIGH-RESOLUTION MATERIAL DATABASE")
        print("=" * 60)
        
        for material, properties in self.materials.items():
            temp_range = properties['temp_range']
            print(f"\n{material.upper().replace('_', ' ')}:")
            print(f"  Temperature range: {temp_range.min():.0f} - {temp_range.max():.0f} K")
            print(f"  Density: {properties.get('density', 'Variable')} kg/m³")
            
            # Show thermal conductivity range
            if 'thermal_conductivity' in properties:
                k_values = properties['thermal_conductivity']
                print(f"  Thermal conductivity: {k_values.min():.3f} - {k_values.max():.3f} W/(m·K)")
            
            # Show specific heat range
            if 'specific_heat' in properties:
                cp_values = properties['specific_heat']
                print(f"  Specific heat: {cp_values.min():.0f} - {cp_values.max():.0f} J/(kg·K)")

# Create global instance
MATERIALS_DB = TubeFurnaceMaterials()

if __name__ == "__main__":
    # Test the materials database
    materials = TubeFurnaceMaterials()
    materials.get_material_summary()

    # Test interpolation
    test_temp = 300
    print(f"\nTest interpolation at: Temperature = {test_temp}K")
    for material in materials.get_all_materials():
        try:
            k, rho, cp = materials.get_thermal_properties(material, test_temp)
            alpha = materials.get_thermal_diffusivity(material, test_temp)
            print(f"{material}: k={k:.3f}, ρ={rho:.1f}, cp={cp:.0f}, α={alpha*1e6:.3f}×10⁻⁶ m²/s")
        except Exception as e:
            print(f"{material}: Error - {e}")