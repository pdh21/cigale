class SKIRTOR2016(object):
    """SKIRTOR (Stalevski et al., 2016) AGN dust torus emission model.

    This class holds the UV-optical data associated with a SKIRTORN AGN model
    (Stalevski et al., 2012, 2016).

    """

    def __init__(self, t, pl, q, oa, R, Mcl, i, wave, disk, dust):
        """Create a new AGN model. The descriptions of the parameters are taken
           directly from https://sites.google.com/site/skirtorus/sed-library.

        Parameters
        ----------
        t: float
            average edge-on optical depth at 9.7 micron; the actual one along
            the line of sight may vary depending on the clumps distribution
        pl: float
            power-law exponent that sets radial gradient of dust density
        q: float
            index that sets dust density gradient with polar angle
        oa: float
            angle measured between the equatorial plan and edge of the torus.
            Half-opening angle of the dust-free cone is 90-oa
        R: float
            ratio of outer to inner radius, R_out/R_in
        Mcl: float
            fraction of total dust mass inside clumps. 0.97 means 97% of total
            mass is inside the clumps and 3% in the interclump dust.
        i: float
            inclination, i.e. viewing angle, i.e. position of the instrument
            w.r.t. the AGN axis. i=0: face-on, type 1 view; i=90: edge-on, type
            2 view.
        wave: array of float
            Wavelength grid in nm.
        disk: array of flaot
            Luminosity of the accretion disk in W/nm
        dust: array of float
            Luminosity of the dust in W/nm

        """

        self.t = t
        self.pl = pl
        self.q = q
        self.oa = oa
        self.R = R
        self.Mcl = Mcl
        self.i = i
        self.wave = wave
        self.disk = disk
        self.dust = dust
