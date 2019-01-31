# -*- coding: utf-8 -*-
# Copyright (C) 2014 Médéric Boquien
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Médéric Boquien

"""
Various utility functions for pcigale analysis modules
"""
import multiprocessing as mp
import time


def nothread():
    """Some libraries such as Intel's MKL have automatic threading. This is
    good when having only one process. However we already do our own
    parallelisation. The additional threads created by the MKL increase in
    excess the pressure on the CPU and on the RAM slowing everything down.

    Parameters
    ----------
    None

    Returns
    -------
    None

    """
    try:
        import mkl
        mkl.set_num_threads(1)
    except ImportError:
        pass


class Counter:
    """Class to count the number of models computers/objects analysed. It has
    two internal counters. One is internal to the process and is incremented at
    each iteration. The other one is global is is only incremented
    periodically. The fundamental reason is that a lock is needed to increment
    the global value. When using many cores this can strongly degrade the
    performance. Similarly printing the number of iterations achieved and at
    what speed is only done periodically. For practical reasons the printing
    frequency has to be a multiple of the incrementation frequency of the
    global counter.
    """

    def __init__(self, nmodels, freq_inc=1, freq_print=1):
        if freq_print % freq_inc != 0:
            raise ValueError("The printing frequency must be a multiple of "
                             "the increment frequency.")
        self.nmodels = nmodels
        self.freq_inc = freq_inc
        self.freq_print = freq_print
        self.global_counter = mp.Value('i', 0)
        self.proc_counter = 0
        self.t0 = time.time()

    def inc(self):
        self.proc_counter += 1
        if self.proc_counter % self.freq_inc == 0:
            with self.global_counter.get_lock():
                self.global_counter.value += self.freq_inc
                n = self.global_counter.value
            if n % self.freq_print == 0:
                self.pprint(n)

    def pprint(self, n):
        dt = time.time() - self.t0
        print("{}/{} performed in {:.1f} seconds ({:.1f}/s)".
              format(n, self.nmodels, dt, n / dt),
              end="\n" if n == self.nmodels else "\r")
