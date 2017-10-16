# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Authors: Yannick Roehlly, Médéric Boquien, Laure Ciesla

"""
This script is used the build pcigale internal database.

"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import glob
import io
import itertools
import numpy as np
from scipy import interpolate
import scipy.constants as cst
from astropy.table import Table
from pcigale.data import (Database, Filter, BC03, Fritz2006, DL2014)


def read_bc03_ssp(filename):
    """Read a Bruzual and Charlot 2003 ASCII SSP file

    The ASCII SSP files of Bruzual and Charlot 2003 have se special structure.
    A vector is stored with the number of values followed by the values
    separated by a space (or a carriage return). There are the time vector, 5
    (for Chabrier IMF) or 6 lines (for Salpeter IMF) that we don't care of,
    then the wavelength vector, then the luminosity vectors, each followed by
    a 52 value table, then a bunch of other table of information that are also
    in the *colors files.

    Parameters
    ----------
    filename : string

    Returns
    -------
    time_grid: numpy 1D array of floats
              Vector of the time grid of the SSP in Myr.
    wavelength: numpy 1D array of floats
                Vector of the wavelength grid of the SSP in nm.
    spectra: numpy 2D array of floats
             Array containing the SSP spectra, first axis is the wavelength,
             second one is the time.

    """

    def file_structure_generator():
        """Generator used to identify table lines in the SSP file

        In the SSP file, the vectors are store one next to the other, but
        there are 5 informational lines after the time vector. We use this
        generator to the if we are on lines to read or not.
        """
        if "chab" in filename:
            bad_line_number = 5
        else:
            bad_line_number = 6
        yield("data")
        for i in range(bad_line_number):
            yield("bad")
        while True:
            yield("data")

    file_structure = file_structure_generator()
    # Are we in a data line or a bad one.
    what_line = next(file_structure)
    # Variable conting, in reverse order, the number of value still to
    # read for the read vector.
    counter = 0

    time_grid = []
    full_table = []
    tmp_table = []

    with open(filename) as file_:
        # We read the file line by line.
        for line in file_:
            if what_line == "data":
                # If we are in a "data" line, we analyse each number.
                for item in line.split():
                    if counter == 0:
                        # If counter is 0, then we are not reading a vector
                        # and the first number is the length of the next
                        # vector.
                        counter = int(item)
                    else:
                        # If counter > 0, we are currently reading a vector.
                        tmp_table.append(float(item))
                        counter -= 1
                        if counter == 0:
                            # We reached the end of the vector. If we have not
                            # yet store the time grid (the first table) we are
                            # currently reading it.
                            if time_grid == []:
                                time_grid = tmp_table[:]
                            # Else, we store the vector in the full table,
                            # only if its length is superior to 250 to get rid
                            # of the 52 item unknown vector and the 221 (time
                            # grid length) item vectors at the end of the
                            # file.
                            elif len(tmp_table) > 250:
                                full_table.append(tmp_table[:])

                            tmp_table = []

            # If at the end of a line, we have finished reading a vector, it's
            # time to change to the next structure context.
            if counter == 0:
                what_line = next(file_structure)

    # The time grid is in year, we want Myr.
    time_grid = np.array(time_grid, dtype=float)
    time_grid *= 1.e-6

    # The first "long" vector encountered is the wavelength grid. The value
    # are in Ångström, we convert it to nano-meter.
    wavelength = np.array(full_table.pop(0), dtype=float)
    wavelength *= 0.1

    # The luminosities are in Solar luminosity (3.826.10^33 ergs.s-1) per
    # Ångström, we convert it to W/nm.
    luminosity = np.array(full_table, dtype=float)
    luminosity *= 3.826e27
    # Transposition to have the time in the second axis.
    luminosity = luminosity.transpose()

    # In the SSP, the time grid begins at 0, but not in the *colors file, so
    # we remove t=0 from the SSP.
    return time_grid[1:], wavelength, luminosity[:, 1:]


def build_filters(base):
    filters = []
    filters_dir = os.path.join(os.path.dirname(__file__), 'filters/')
    for filter_file in glob.glob(filters_dir + '*.dat'):
        with open(filter_file, 'r') as filter_file_read:
            filter_name = filter_file_read.readline().strip('# \n\t')
            filter_type = filter_file_read.readline().strip('# \n\t')
            filter_description = filter_file_read.readline().strip('# \n\t')
        filter_table = np.genfromtxt(filter_file)
        # The table is transposed to have table[0] containing the wavelength
        # and table[1] containing the transmission.
        filter_table = filter_table.transpose()

        # We convert the wavelength from Å to nm.
        filter_table[0] *= 0.1

        # We convert to energy if needed
        if filter_type == 'photon':
            filter_table[1] *= filter_table[0]
        elif filter_type != 'energy':
            raise ValueError("Filter transmission type can only be "
                             "'energy' or 'photon'.")

        print("Importing %s... (%s points)" % (filter_name,
                                               filter_table.shape[1]))

        new_filter = Filter(filter_name, filter_description, filter_table)

        # We normalise the filter and compute the effective wavelength.
        # If the filter is a pseudo-filter used to compute line fluxes, it
        # should not be normalised.
        if not filter_name.startswith('PSEUDO'):
            new_filter.normalise()
        else:
            new_filter.effective_wavelength = np.mean(
                filter_table[0][filter_table[1] > 0]
            )
        filters.append(new_filter)

    base.add_filters(filters)


def build_bc2003(base):
    bc03_dir = os.path.join(os.path.dirname(__file__), 'bc03//')

    # Time grid (1 Myr to 14 Gyr with 1 Myr step)
    time_grid = np.arange(1, 14000)

    # Metallicities associated to each key
    metallicity = {
        "m22": 0.0001,
        "m32": 0.0004,
        "m42": 0.004,
        "m52": 0.008,
        "m62": 0.02,
        "m72": 0.05
    }

    for key, imf in itertools.product(metallicity, ["salp", "chab"]):
        base_filename = bc03_dir + "bc2003_lr_" + key + "_" + imf + "_ssp"
        ssp_filename = base_filename + ".ised_ASCII"
        color3_filename = base_filename + ".3color"
        color4_filename = base_filename + ".4color"

        print("Importing %s..." % base_filename)

        # Read the desired information from the color files
        color_table = []
        color3_table = np.genfromtxt(color3_filename).transpose()
        color4_table = np.genfromtxt(color4_filename).transpose()
        color_table.append(color4_table[6])        # Mstar
        color_table.append(color4_table[7])        # Mgas
        color_table.append(10 ** color3_table[5])  # NLy
        color_table.append(color3_table[1])        # B4000
        color_table.append(color3_table[2])        # B4_VN
        color_table.append(color3_table[3])        # B4_SDSS
        color_table.append(color3_table[4])        # B(912)

        color_table = np.array(color_table)

        ssp_time, ssp_wave, ssp_lumin = read_bc03_ssp(ssp_filename)

        # Regrid the SSP data to the evenly spaced time grid.
        color_table = interpolate.interp1d(ssp_time, color_table)(time_grid)
        ssp_lumin = interpolate.interp1d(ssp_time,
                                         ssp_lumin)(time_grid)

        # To avoid the creation of waves when interpolating, we refine the grid
        # beyond 10 μm following a log scale in wavelength. The interpolation
        # is also done in log space as the spectrum is power-law-like
        ssp_wave_resamp = np.around(np.logspace(np.log10(10000),
                                                np.log10(160000), 50))
        argmin = np.argmin(10000.-ssp_wave > 0)-1
        ssp_lumin_resamp = 10.**interpolate.interp1d(
                                    np.log10(ssp_wave[argmin:]),
                                    np.log10(ssp_lumin[argmin:, :]),
                                    assume_sorted=True,
                                    axis=0)(np.log10(ssp_wave_resamp))

        ssp_wave = np.hstack([ssp_wave[:argmin+1], ssp_wave_resamp])
        ssp_lumin = np.vstack([ssp_lumin[:argmin+1, :], ssp_lumin_resamp])

        base.add_bc03(BC03(
            imf,
            metallicity[key],
            time_grid,
            ssp_wave,
            color_table,
            ssp_lumin
        ))


def build_dl2014(base):
    models = []
    dl2014_dir = os.path.join(os.path.dirname(__file__), 'dl2014/')

    qpah = {"000": 0.47, "010": 1.12, "020": 1.77, "030": 2.50, "040": 3.19,
            "050": 3.90, "060": 4.58, "070": 5.26, "080": 5.95, "090": 6.63,
            "100": 7.32}

    uminimum = ["0.100", "0.120", "0.150", "0.170", "0.200", "0.250", "0.300",
                "0.350", "0.400", "0.500", "0.600", "0.700", "0.800", "1.000",
                "1.200", "1.500", "1.700", "2.000", "2.500", "3.000", "3.500",
                "4.000", "5.000", "6.000", "7.000", "8.000", "10.00", "12.00",
                "15.00", "17.00", "20.00", "25.00", "30.00", "35.00", "40.00",
                "50.00"]

    alpha = ["1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8",
             "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
             "2.8", "2.9", "3.0"]

    # Mdust/MH used to retrieve the dust mass as models as given per atom of H
    MdMH = {"000": 0.0100, "010": 0.0100, "020": 0.0101, "030": 0.0102,
            "040": 0.0102, "050": 0.0103, "060": 0.0104, "070": 0.0105,
            "080": 0.0106, "090": 0.0107, "100": 0.0108}

    # Here we obtain the wavelength beforehand to avoid reading it each time.
    datafile = open(dl2014_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                    .format(uminimum[0], uminimum[0], "000"))

    data = "".join(datafile.readlines()[-1001:])
    datafile.close()

    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    # For some reason wavelengths are decreasing in the model files
    wave = wave[::-1]
    # We convert wavelengths from μm to nm
    wave *= 1000.

    # Conversion factor from Jy cm² sr¯¹ H¯¹ to W nm¯¹ (kg of H)¯¹
    conv = 4. * np.pi * 1e-30 / (cst.m_p+cst.m_e) * cst.c / (wave*wave) * 1e9

    for model in sorted(qpah.keys()):
        for umin in uminimum:
            filename = (dl2014_dir + "U{}_{}_MW3.1_{}/spec_1.0.dat"
                        .format(umin, umin, model))
            print("Importing {}...".format(filename))
            with open(filename) as datafile:
                data = "".join(datafile.readlines()[-1001:])
            lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
            # For some reason fluxes are decreasing in the model files
            lumin = lumin[::-1]

            # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
            lumin *= conv/MdMH[model]

            models.append(DL2014(qpah[model], umin, umin, 1.0, wave, lumin))
            for al in alpha:
                filename = (dl2014_dir + "U{}_1e7_MW3.1_{}/spec_{}.dat"
                            .format(umin, model, al))
                print("Importing {}...".format(filename))
                with open(filename) as datafile:
                    data = "".join(datafile.readlines()[-1001:])
                lumin = np.genfromtxt(io.BytesIO(data.encode()), usecols=(2))
                # For some reason fluxes are decreasing in the model files
                lumin = lumin[::-1]

                # Conversion from Jy cm² sr¯¹ H¯¹to W nm¯¹ (kg of dust)¯¹
                lumin *= conv/MdMH[model]

                models.append(DL2014(qpah[model], umin, 1e7, al, wave, lumin))

    base.add_dl2014(models)

def build_fritz2006(base):
    models = []
    fritz2006_dir = os.path.join(os.path.dirname(__file__), 'fritz2006/')

    # Parameters of Fritz+2006
    psy = [0.001, 10.100, 20.100, 30.100, 40.100, 50.100, 60.100, 70.100,
           80.100, 89.990]  # Viewing angle in degrees
    opening_angle = ["20", "40", "60"]  # Theta = 2*(90 - opening_angle)
    gamma = ["0.0", "2.0", "4.0", "6.0"]
    beta = ["-1.00", "-0.75", "-0.50", "-0.25", "0.00"]
    tau = ["0.1", "0.3", "0.6", "1.0", "2.0", "3.0", "6.0", "10.0"]
    r_ratio = ["10", "30", "60", "100", "150"]

    # Read and convert the wavelength
    datafile = open(fritz2006_dir + "ct{}al{}be{}ta{}rm{}.tot"
                    .format(opening_angle[0], gamma[0], beta[0], tau[0],
                            r_ratio[0]))
    data = "".join(datafile.readlines()[-178:])
    datafile.close()
    wave = np.genfromtxt(io.BytesIO(data.encode()), usecols=(0))
    wave *= 1e3
    # Number of wavelengths: 178; Number of comments lines: 28
    nskip = 28
    blocksize = 178

    iter_params = ((oa, gam, be, ta, rm)
                   for oa in opening_angle
                   for gam in gamma
                   for be in beta
                   for ta in tau
                   for rm in r_ratio)

    for params in iter_params:
        filename = fritz2006_dir + "ct{}al{}be{}ta{}rm{}.tot".format(*params)
        print("Importing {}...".format(filename))
        try:
            datafile = open(filename)
        except IOError:
            continue
        data = datafile.readlines()
        datafile.close()

        for n in range(len(psy)):
            block = data[nskip + blocksize * n + 4 * (n + 1) - 1:
                         nskip + blocksize * (n+1) + 4 * (n + 1) - 1]
            lumin_therm, lumin_scatt, lumin_agn = np.genfromtxt(
                io.BytesIO("".join(block).encode()), usecols=(2, 3, 4),
                unpack=True)
            # Remove NaN
            lumin_therm = np.nan_to_num(lumin_therm)
            lumin_scatt = np.nan_to_num(lumin_scatt)
            lumin_agn = np.nan_to_num(lumin_agn)
            # Conversion from erg/s/microns to W/nm
            lumin_therm *= 1e-4
            lumin_scatt *= 1e-4
            lumin_agn *= 1e-4
            # Normalization of the lumin_therm to 1W
            norm = np.trapz(lumin_therm, x=wave)
            lumin_therm /= norm
            lumin_scatt /= norm
            lumin_agn /= norm

            models.append(Fritz2006(params[4], params[3], params[2],
                                         params[1], params[0], psy[n], wave,
                                         lumin_therm, lumin_scatt, lumin_agn))

    base.add_fritz2006(models)

def build_base():
    base = Database(writable=True)
    base.upgrade_base()

    print('#' * 78)
    print("1- Importing filters...\n")
    build_filters(base)
    print("\nDONE\n")
    print('#' * 78)

    print("3- Importing Bruzual and Charlot 2003 SSP\n")
    build_bc2003(base)
    print("\nDONE\n")
    print('#' * 78)

    print("5- Importing the updated Draine and Li (2007 models)\n")
    build_dl2014(base)
    print("\nDONE\n")
    print('#' * 78)

    print("6- Importing Fritz et al. (2006) models\n")
    build_fritz2006(base)
    print("\nDONE\n")
    print('#' * 78)

    base.session.close_all()


if __name__ == '__main__':
    build_base()
