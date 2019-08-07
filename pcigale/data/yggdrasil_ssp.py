import numpy as np


class Yggdrasil_SSP(object):


    def __init__(self, metallicity, fcov, time_grid, wavelength_grid,
                 info_table, spec_table):

        self.metallicity = metallicity
        self.fcov = fcov
        self.time_grid = time_grid
        self.wavelength_grid = wavelength_grid
        self.info_table = info_table
        self.spec_table = spec_table
