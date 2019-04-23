# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

import numpy as np


class BC03_SSP(object):


    def __init__(self, imf, metallicity, time_grid, wavelength_grid,
                 info_table, spec_table):


        if imf in ['salp', 'chab']:
            self.imf = imf
        else:
            raise ValueError('IMF must be either sal for Salpeter or '
                             'cha for Chabrier.')
        self.metallicity = metallicity
        self.time_grid = time_grid
        self.wavelength_grid = wavelength_grid
        self.info_table = info_table
        self.spec_table = spec_table
