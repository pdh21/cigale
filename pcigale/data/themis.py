# -*- coding: utf-8 -*-
# DustPedia,  http://www.dustpedia.com/
# Author: Angelos Nersesian, Frederic Galliano, Anthony Jones, Pieter De Vis


class THEMIS(object):
    """Jones et al (2017) dust models,

    This class holds the data associated with the Jones et al (2017)
    dust models.

    """

    def __init__(self, qhac, umin, umax, alpha, wave, lumin):
        """Create a new IR model

        Parameters
        ----------
        qhac: float
            Mass fraction of hydrocarbon solids i.e., a-C(:H) smaller than
        1.5 nm, also known as HAC
        umin: float
            Minimum radiation field illuminating the dust
        umax: float
            Maximum radiation field illuminating the dust
        alpha: float
            Powerlaw slope dU/dM∝U¯ᵅ
        wave: array
            Vector of the λ grid used in the templates [nm]
        lumin: array
            Model data in an array containing the luminosity density
        versus the wavelength λ

        """

        self.qhac = qhac
        self.umin = umin
        self.umax = umax
        self.alpha = alpha
        self.wave = wave
        self.lumin = lumin
