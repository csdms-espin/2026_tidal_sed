#!/usr/env/python

"""
Calculate bed level evolution based on a tidal flow velocity field using the approach of Mariotti (2018)

"""

import numpy as np
from landlab import Component
from landlab.grid.mappers import map_mean_of_link_nodes_to_link

class BedLevelEvolution(Component):
    """Component to calculate bed level evolution based on a flow velocity field.

    This component computes the rate of change of bed elevation driven
    by erosion, deposition, diffusion, and relative sea level rise. 
    Erosion is calculated from the bed shear stress associated with flood-
    and ebb-tide flow velocities produced by the TidalFlowCalculator component; 
    deposition uses a settling velocity/concentration formulation. 
    Bed elevation change is integrated as

        dz/dt = diffusivity * grad^2(z) + relative_sea_level_rise
                + (D - E) / rho_sediment

    where D and E are deposition and erosion fluxes.
    
    References
    ----------
    Mariotti, G. (2018) Marsh channel morphological response to sea level rise
    and sediment supply. Estuarine, Coastal and Shelf Science, 209, 89--101,
    https://doi.org/10.1016/j.ecss.2018.05.016.   
    
    """

    _name = "BedEvolution"

    _unit_agnostic = False

    _info = {
        "topographic__elevation": {
            "dtype": float,
            "intent": "inout",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Land surface topographic elevation",
        },
        "mean_water__depth": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m",
            "mapping": "node",
            "doc": "Mean water depth over a tidal cycle",
        },
        "flood_tide_flow__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "link",
            "doc": "Average horizontal flow velocity during flood tide",
        },
        "ebb_tide_flow__velocity": {
            "dtype": float,
            "intent": "in",
            "optional": False,
            "units": "m/s",
            "mapping": "link",
            "doc": "Average horizontal flow velocity during ebb tide",
        },
    }

    
    def __init__(
        self,
        grid,
        diffusivity = 3.75,
        relative_sea_level_rise = 0,
        settling_velocity = 0.03,
        sediment_concentration = 0.1,
        critical_shear_stress = 0.2,
        erodibility_coefficient = 1e-3,
        mannings_n = 0.035,
        rho_water = 1025,
        rho_sediment = 2650,
        g = 9.81,
    ):
        """Initialize the BedLevelEvolution model.
        
        Parameters
        ----------
        grid : ModelGrid
            Landlab ModelGrid object

    v: velocity vectors (on links) - output from TidalFlowCalculator component

  

    
            
        diffusivity : float, optional
            Diffusion coefficient. Defaults to 3.65 (Kirwan and Murray, 2007) -> now 3.75?
        relative_sea_level_rise : float, optional
            Relative sea level rise. Defaults to 0, no sea level change.
        settling_velocity : float, optional
            Settling velocity of sediment. Defaults to 0.03 [m/s] for fine sand.
        sediment_concentration : float, optional
            Average sediment concentration of the water. Defaults to 0.1 [kg/m3] for tidal waters.
        critical_shear_stress : float, optional
            Critical shear stress for erosion of the bed. Defauls to 0.2 [Pa] for fine sand.
        erodibility_coefficient : float, optional
            Erodibility coefficient for the bed. Defaults to 10^-3 for fine sand.
        mannings_n : float, optional
            Manning’s Roughness Coefficient for open channel flow. Defaults to 0.035 for open water.
        rho_water : float, optional
            Density of water. Defaults to 1025 [kg/m3].
        rho_sediment : float, optional
            Density of sediment. Defaults to 2650 [kg/m3].
        g : float, optional
            Gravitational constant. Defaults to 9.81 [m/s2].
            
        """
        super().__init__(grid)

        self._diffusivity = diffusivity
        self._relative_sea_level_rise = relative_sea_level_rise
        self._settling_velocity = settling_velocity
        self._sediment_concentration = sediment_concentration
        self._critical_shear_stress = critical_shear_stress
        self._erodibility_coefficient = erodibility_coefficient
        self._mannings_n = mannings_n #could we include this instead by using the grid-dependent roughness?
        self._rho_water = rho_water
        self._rho_sediment = rho_sediment
        self._g = g
    
        self._elev = grid.at_node["topographic__elevation"]

    def calc_deposition(self) -> float:
        """ Compute deposition.
        
        Returns
        -------
        float
            Deposition flux [kg/m2/s].
        """
        return self._settling_velocity * self._sediment_concentration


    def calc_shear(self, flow_velocity):
        """ Compute bed shear stress at flow links.
        
        Returns
        -------
        float
            Shear stress [].
        """
        
        # water depth is transferred from nodes to links, velocity is on links
        depth_at_links = map_mean_of_link_nodes_to_link(self.grid, "mean_water__depth")
       
        return (self._rho_water * self._g * (self._mannings_n ** 2) * (depth_at_links ** (-1 / 3)) * np.sign(flow_velocity) * (flow_velocity ** 2))

    def calc_erosion(self, flow_velocity):
        """ Compute erosion flux at links.

        Parameters
        ------
        flow_velocity : ndarray
            Flow velocity field across the grid object. Uses grid object output from TidalFlowCalculator
            with "flood_tide_flow__velocity" and "ebb_tide_flow__velocity" 
            for flood and ebb flow velocities, respectively.
    
        Returns
        -------
        ndarray of float
            Erosion flux at links.
        
        """
        shear_ratio = self.calc_shear(flow_velocity) / self._critical_shear_stress

        return self._erodibility_coefficient * (np.sqrt(1 + shear_ratio ** 2) - 1)
        
    def calc_bed_level_change(self):
        """Compute the rate of bed level change at nodes.

        Returns
        -------
        ndarray of float
            Rate of bed level change, dz / dt.
        """
        deposition = self.calc_deposition()
        flood_erosion = self.calc_erosion(self.grid.at_link["flood_tide_flow__velocity"])
        ebb_erosion = self.calc_erosion(self.grid.at_link["ebb_tide_flow__velocity"])
        erosion = (flood_erosion + ebb_erosion ) / 2
        erosion_at_nodes = self.grid.map_sum_of_inlinks_to_node(erosion) + self.grid.map_sum_of_outlinks_to_node(erosion)

        # compute bed level change
        dz_dt = (
            self.grid.calc_flux_div_at_node(
                self._diffusivity * self.grid.calc_grad_at_link(self._elev)
            )
            + self._relative_sea_level_rise
            + ((deposition - erosion_at_nodes) / self._rho_sediment)
        )
        return dz_dt

    def run_one_step(self, dt = 1.0):
        """Compute bed level change and update elevation."""
        dz_dt = self.calc_bed_level_change()
        self._elev[:] += dz_dt * dt
        return dz_dt




        






        