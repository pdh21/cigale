# -*- coding: utf-8 -*-
# Copyright (C) 2012, 2013 Centre de données Astrophysiques de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly

import argparse
import multiprocessing as mp
import sys

from .session.configuration import Configuration
from .analysis_modules import get_module
from .managers.parameters import ParametersManager

__version__ = "0.10.0"


def init(config):
    """Create a blank configuration file.
    """
    config.create_blank_conf()
    print("The initial configuration file was created. Please complete it "
          "with the data file name and the pcigale modules to use.")


def genconf(config):
    """Generate the full configuration.
    """
    config.generate_conf()
    print("The configuration file has been updated. Please complete the "
          "various module parameters and the data file columns to use in "
          "the analysis.")


def check(config):
    """Check the configuration.
    """
    # TODO: Check if all the parameters that don't have default values are
    # given for each module.
    configuration = config.configuration

    if configuration:
        print("With this configuration cigale will compute {} "
              "models.".format(ParametersManager(configuration).size))


def run(config):
    """Run the analysis.
    """
    configuration = config.configuration

    if configuration:
        analysis_module = get_module(configuration['analysis_method'])
        analysis_module.process(configuration)


def main():
    if sys.version_info[:2] < (3, 5):
        raise Exception("Python {}.{} is unsupported. Please upgrade to "
                        "Python 3.5 or later.".format(*sys.version_info[:2]))
    if sys.version_info[:2] < (3, 6):
        print("Python {}.{} detected. For better performance we recommend "
              "Python 3.6 or later.".format(*sys.version_info[:2]))

    # We set the sub processes start method to spawn because it solves
    # deadlocks when a library cannot handle being used on two sides of a
    # forked process. This happens on modern Macs with the Accelerate library
    # for instance. On Linux we should be pretty safe with a fork, which allows
    # to start processes much more rapidly.
    if sys.platform.startswith('linux'):
        mp.set_start_method('fork')
    else:
        mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--conf-file', dest='config_file',
                        help="Alternative configuration file to use.")

    subparsers = parser.add_subparsers(help="List of commands")

    init_parser = subparsers.add_parser('init', help=init.__doc__)
    init_parser.set_defaults(parser='init')

    genconf_parser = subparsers.add_parser('genconf', help=genconf.__doc__)
    genconf_parser.set_defaults(parser='genconf')

    check_parser = subparsers.add_parser('check', help=check.__doc__)
    check_parser.set_defaults(parser='check')

    run_parser = subparsers.add_parser('run', help=run.__doc__)
    run_parser.set_defaults(parser='run')

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        args = parser.parse_args()

        if args.config_file:
            config = Configuration(args.config_file)
        else:
            config = Configuration()

        if args.parser == 'init':
            init(config)
        elif args.parser == 'genconf':
            genconf(config)
        elif args.parser == 'check':
            check(config)
        elif args.parser == 'run':
            run(config)
