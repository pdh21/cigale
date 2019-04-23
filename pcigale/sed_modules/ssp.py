"""
Simple module to reduce the SFH to a single SSP
===========================================================

This module implements a star formation history (SFH) through a single SSP.

"""

from collections import OrderedDict

import numpy as np

from . import SedModule


class SSP(SedModule):
    """Instantaneous burst corresponding to a model-provided SSP

    This module sets the SED star formation history (SFH) as a single stellar
    population

    """

    parameter_list = OrderedDict([
        ("index", (
            "cigale_list(dtype=int, minvalue=0)",
            "Index of the SSP to use.",
            0
        ))
    ])

    def _init_code(self):
        self.index = int(self.parameters["index"])

    def process(self, sed):
        """Add a double decreasing exponential Star Formation History.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """

        sed.add_module(self.name, self.parameters)

        # Add the sfh and the output parameters to the SED.
        sed.sfh = np.array([0.])
        sed.add_info("ssp.index", self.index)


# SedModule to be returned by get_module
Module = SSP
