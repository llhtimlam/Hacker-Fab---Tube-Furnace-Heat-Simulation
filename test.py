import numpy as np
import config
from materials import TubeFurnaceMaterials
from mesh import TubeFurnaceMesh
from solver import TubeFurnaceHeatSolver
import pandas as pd
class Test:
    def __init__(self):
        self.materials = TubeFurnaceMaterials()
        self.mesh = TubeFurnaceMesh()
        self.solver = TubeFurnaceHeatSolver()
    def runtest(self):
        k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.mesh.material_map_cylindrical_centered,
            298.15
        )
        print(k_matrix)
    def runtest2(self):
        #print(self.mesh.r_nodes)
        #print(self.mesh.r_faces)
        #print(self.mesh.r_centers)
        #np.savetxt("r_mapping.csv", self.mesh.material_map_cylindrical_centered)
        #np.savetxt("r_nodes.csv", self.mesh.r_nodes)
        #np.savetxt("r_centers.csv", self.mesh.r_centers)
        #np.savetxt("r_faces.csv", self.mesh.r_faces)
        #arr=self.mesh.material_map_cylindrical_centered
        #print(arr[2])
        print(self.mesh.r_nodes.shape)
        print(self.mesh.r_faces.shape)
        print(self.mesh.r_centers.shape)
        print(self.mesh.starting_index)
        if False:
            print(self.mesh.glass_kanthal_cement_interface)
            print(self.mesh.kanthal_cement_interface)
            print(self.mesh.cement_ceramic_interface)
            print(self.mesh.ceramic_reflective_interface)
            print(self.mesh.reflective_outer_interface)
            #print(self.mesh.cold_heat_interface_front)
            #print(self.mesh.cold_heat_interface_back)
        if False:
            print(self.mesh.kanthal_r_position_start)
            print(self.mesh.kanthal_r_position_end)
            print(self.mesh.kanthal_z_position_start)
            print(self.mesh.kanthal_z_position_end)

        if True:
            print(self.mesh.sample_radius)
            print(self.mesh.glass_outer_radius)
            print(self.mesh.kanthal_outer_radius)
            print(self.mesh.cement_outer_radius)
            print(self.mesh.ceramic_outer_radius)
            print(self.mesh.reflective_outer_radius)
            print(self.mesh.inner_glass_tube_area)
            print(self.mesh.sample_air_space_volume)
            print(self.mesh.cubic_inner_surface)
            print(self.mesh.cubic_outer_surface)
    def runtest3(self):
        #np.savetxt("r_scenters.csv", self.solver.r_centers)
        #np.savetxt("r_sfaces.csv", self.solver.r_faces)
        #print(self.solver.r_centers.shape)
        #print(self.solver.r_faces.shape)
        #print(self.solver.dr_centers.shape)
        if False:
            pd.DataFrame(self.solver.dr_grid).to_csv("dr_grid.csv", index=False)
            pd.DataFrame(self.solver.dr_centers).to_csv("dr_centers.csv", index=False)
            pd.DataFrame(self.solver.dz_centers).to_csv("dz_centers.csv", index=False)
            pd.DataFrame(self.solver.dr_grid).to_csv("dr_grid.csv", index=False)
            pd.DataFrame(self.solver.dz_grid).to_csv("dz_grid.csv", index=False)
            pd.DataFrame(self.solver.dr1_grid).to_csv("dr1_grid.csv", index=False)
            pd.DataFrame(self.solver.dr2_grid).to_csv("dr2_grid.csv", index=False)
            pd.DataFrame(self.solver.dz1_grid).to_csv("dz1_grid.csv", index=False)
            pd.DataFrame(self.solver.dz2_grid).to_csv("dz2_grid.csv", index=False)
        if True:
            
            k_matrix, rho_matrix, cp_matrix = self.materials.get_thermal_properties_vec(
            self.materials,
            self.material_map_cylindrical_centered,
            298.15
            )
            pd.DataFrame(k_matrix).to_csv("k_matrix.csv", index=False)
            pd.DataFrame(self.solver.dr1_grid).to_csv("dr1_grid.csv", index=False)
            pd.DataFrame(self.solver.dr2_grid).to_csv("dr2_grid.csv", index=False)
            pd.DataFrame(self.solver.r_faces).to_csv("r_.csv", index=False)
            pd.DataFrame(self.solver.r_centers).to_csv("r_centers.csv", index=False)
            arr = self.solver.harmonic_mean_nonuniform(k_matrix[:-1, :], k_matrix[1:, :], self.solver.dr1_grid, self.solver.dr2_grid)
            pd.DataFrame(arr).to_csv("k_harmonic_mean.csv", index=False)

if __name__ == "__main__":
    # Create an instance of the Test class
    test_instance = Test()
    
    # Call the method from the instance
    test_instance.runtest2()