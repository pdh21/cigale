# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_almost_equal

from pcigale.analysis_modules import adjust_data


def test_adjust_data():

    fluxes = np.array([10., 0., -9999, 10., 10., 10., -10, -10., -10])
    errors = np.array([0.1, 0.1, 0.1, -9999., 0, -0.1, 0.1, 0., -0.1])

    adjusted_fluxes, adjusted_errors = adjust_data(
        fluxes, errors, tolerance=1.e-12, lim_flag=False, default_error=0.1,
        systematic_deviation=0.1)

    # (10, 0.1) is ok. We just add the systematic deviation.
    assert_almost_equal(adjusted_fluxes[0], 10.)
    assert_almost_equal(adjusted_errors[0], np.sqrt(0.1**2+1))

    # (0., 0.1) is is invalid.
    assert(np.isnan([adjusted_fluxes[1], adjusted_errors[1]]).all())

    # (-9999, 0.1) is invalid
    assert(np.isnan([adjusted_fluxes[2], adjusted_errors[2]]).all())

    # (10., -9999.) we use the default error and add the systematic deviation.
    assert_almost_equal(adjusted_fluxes[3], 10.)
    assert_almost_equal(adjusted_errors[3], np.sqrt(2))

    # (10., 0.) With error to 0, we use the default error and add the
    # systematic deviation.
    assert_almost_equal(adjusted_fluxes[4], 10.)
    assert_almost_equal(adjusted_errors[4], np.sqrt(2))

    # (10., -0.1) Negative error is invalid, we use the default error and add
    # the systematic deviation.
    assert_almost_equal(adjusted_fluxes[5], 10.)
    assert_almost_equal(adjusted_errors[5], np.sqrt(2))

    # With lim_flag set to False, negative fluxes are invalid values
    assert(np.isnan([adjusted_fluxes[6], adjusted_errors[6]]).all())
    assert(np.isnan([adjusted_fluxes[7], adjusted_errors[7]]).all())
    assert(np.isnan([adjusted_fluxes[8], adjusted_errors[8]]).all())

    # With lim_flag set to True
    adjusted_fluxes, adjusted_errors = adjust_data(
        fluxes, errors, tolerance=1.e-12, lim_flag=True, default_error=0.1,
        systematic_deviation=0.1)

    # (-10, 0.1) Upper limit, we add the systematic deviation to the error.
    assert_almost_equal(adjusted_fluxes[6], -10.)
    assert_almost_equal(adjusted_errors[6], np.sqrt(0.1**2+1))

    # (-10., 0.) Upper limit, we use the default error and add the systematic
    # deviation.
    assert_almost_equal(adjusted_fluxes[7], -10.)
    assert_almost_equal(adjusted_errors[7], np.sqrt(2))

    # (-10., -9999) Upper limit, we use the default error and add
    # the systematic deviation.
    assert_almost_equal(adjusted_fluxes[8], -10.)
    assert_almost_equal(adjusted_errors[8], np.sqrt(2))


