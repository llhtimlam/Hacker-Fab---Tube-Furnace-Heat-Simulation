import numpy as np
import config
from materials import TubeFurnaceMaterials
from mesh import TubeFurnaceMesh

class Test:
    def __init__(self):
        self.materials = TubeFurnaceMaterials()
        self.mesh = TubeFurnaceMesh()
    def run_simulation(self):
        k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.mesh.material_map_cylindrical_centered,
            298.15
        )
        print(k_matrix)


if __name__ == "__main__":
    # Create an instance of the Test class
    test_instance = Test()
    
    # Call the method from the instance
    test_instance.run_simulation()