data_file = string()
parameters_file = string()
sed_modules = cigale_string_list()
analysis_method = string()
cores = integer(min=1)
bands = cigale_string_list()
[sed_modules_params]
  [[sfhdelayedplusExpburst]]
    tau_main = cigale_list()
    tau_burst = cigale_list()
    f_burst = cigale_list(minvalue=0., maxvalue=0.9999)
    age = cigale_list(dtype=int, minvalue=0.)
    burst_age = cigale_list(dtype=int, minvalue=1.)
    sfr_0 = float(min=0)
    normalise = boolean()
  [[bc03]]
    imf = cigale_list(dtype=int, options=0. & 1.)
    metallicity = cigale_list(options=0.0001 & 0.0004 & 0.004 & 0.008 & 0.02 & 0.05)
    separation_age = cigale_list(dtype=int, minvalue=0)
  [[dustatt_2powerlaws]]
    Av_BC = cigale_list(minvalue=0)
    slope_BC = cigale_list()
    BC_to_ISM_factor = cigale_list(minvalue=0., maxvalue=1.)
    slope_ISM = cigale_list()
    filters = string()
  [[lyc_absorption]]
    f_esc = cigale_list(minvalue=0., maxvalue=1.)
    f_dust = cigale_list(minvalue=0., maxvalue=1.)
  [[dl2014]]
    qpah = cigale_list(options=0.47 & 1.12 & 1.77 & 2.50 & 3.19 & 3.90 & 4.58 & 5.26 & 5.95 & 6.63 & 7.32)
    umin = cigale_list(options=0.10 & 0.12 & 0.15 & 0.17 & 0.20 & 0.25 & 0.30 & 0.35 & 0.40 & 0.50 & 0.60 & 0.70 & 0.80 & 1.00 & 1.20 & 1.50 & 1.70 & 2.00 & 2.50 & 3.00 & 3.50 & 4.00 & 5.00 & 6.00 & 7.00 & 8.00 & 10.00 & 12.00 & 15.00 & 17.00 & 20.00 & 25.00 & 30.00 & 35.00 & 40.00 & 50.00)
    alpha = cigale_list(options=1.0 & 1.1 & 1.2 & 1.3 & 1.4 & 1.5 & 1.6 & 1.7 & 1.8 & 1.9 & 2.0 & 2.1 & 2.2 & 2.3 & 2.4 & 2.5 & 2.6 & 2.7 & 2.8 & 2.9 & 3.0)
    gamma = cigale_list(minvalue=0., maxvalue=1.)
  [[fritz2006]]
    r_ratio = cigale_list(options=10. & 30. & 60. & 100. & 150.)
    tau = cigale_list(options=0.1 & 0.3 & 0.6 & 1.0 & 2.0 & 3.0 & 6.0 & 10.0)
    beta = cigale_list(options=-1.00 & -0.75 & -0.50 & -0.25 & 0.00)
    gamma = cigale_list(options=0.0 & 2.0 & 4.0 & 6.0)
    opening_angle = cigale_list(options=60. & 100. & 140.)
    psy = cigale_list(options=0.001 & 10.100 & 20.100 & 30.100 & 40.100 & 50.100 & 60.100 & 70.100 & 80.100 & 89.990)
    fracAGN = cigale_list(minvalue=0., maxvalue=1.)
  [[redshifting]]
    redshift = cigale_list(minvalue=0.)
[analysis_params]
  variables = cigale_string_list()
  save_sed = boolean()
  blocks = integer(min=1)
