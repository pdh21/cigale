# -*- coding: utf-8 -*-
# Copyright (C) 2015 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Denis Burgarella

"""
Module that estimates observed fluxes
==========================================================================

This module estimates fluxes from the observed spectrum in order to analyse
them.

This module will be the last one (after redshifting) to account
for all the physical processes at play to build the received total emission.

"""

from collections import OrderedDict

from . import SedModule


class Fluxes(SedModule):
    """Measure the final fluxes to allow cigale to analyze them without
       fitting then.

    This module does not need any input.
    """

    parameter_list = OrderedDict([
        ("filter_list", (
            "string()",
            "Filters for which the flux will be computed. You can give "
            "several filter names separated by a & (don't use commas).",
            ""
        ))
    ])

    def process(self, sed):
        """Computes the parameters for each model.

        Parameters
        ----------
        sed: pcigale.sed.SED object

        """
        # Retrieve the final computed SED using all the previous modules
        # including the IGM and the redshifting. In other words,
        # this module must be the last one. Note that it does require
        # an SFH and an SSP module but nothing else (except redshifting)

        # Computation of fluxes
        filter_list = [item.strip() for item in
                       self.parameters["filter_list"].split("&")
                       if item.strip() != '']

        for filter_ in filter_list:
            sed.add_info(
                f"param.{filter_}",
                sed.compute_fnu(filter_),
                True
            )


# SedModule to be returned by get_module
Module = Fluxes
