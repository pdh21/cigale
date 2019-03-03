# -*- coding: utf-8 -*-
# Copyright (C) 2013 Centre de données Astrophysiques de Marseille
# Copyright (C) 2013-2014 Yannick Roehlly
# Copyright (C) 2013 Institute of Astronomy
# Copyright (C) 2014 Laboratoire d'Astrophysique de Marseille
# Licensed under the CeCILL-v2 licence - see Licence_CeCILL_V2-en.txt
# Author: Yannick Roehlly, Médéric Boquien & Denis Burgarella

import argparse
import sys
from os import path
import multiprocessing as mp

from pcigale.session.configuration import Configuration
from .plot_types.chi2 import chi2 as chi2_action
from .plot_types.pdf import pdf as pdf_action
from .plot_types.sed import sed as sed_action
from .plot_types.mock import mock as mock_action

__version__ = "0.2-alpha"


def main():

    if sys.version_info[:2] >= (3, 4):
        mp.set_start_method('spawn')
    else:
        print("Could not set the multiprocessing start method to spawn. If "
              "you encounter a deadlock, please upgrade to Python≥3.4.")

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--conf-file', dest='config_file',
                        help="Alternative configuration file to use.")

    subparsers = parser.add_subparsers(help="List of commands")

    pdf_parser = subparsers.add_parser('pdf', help=pdf_action.__doc__)
    pdf_parser.add_argument('--outdir', dest='outdir', default='out')
    pdf_parser.set_defaults(parser='pdf')

    chi2_parser = subparsers.add_parser('chi2', help=chi2_action.__doc__)
    chi2_parser.add_argument('--outdir', dest='outdir', default='out')
    chi2_parser.set_defaults(parser='chi2')

    sed_parser = subparsers.add_parser('sed', help=sed_action.__doc__)
    sed_parser.add_argument('--type', default='mJy')
    sed_parser.add_argument('--nologo', action='store_true')
    sed_parser.add_argument('--outdir', dest='outdir', default='out')
    sed_parser.set_defaults(parser='sed')

    mock_parser = subparsers.add_parser('mock', help=mock_action.__doc__)
    mock_parser.add_argument('--nologo', action='store_true')
    mock_parser.add_argument('--outdir', dest='outdir', default='out')
    mock_parser.set_defaults(parser='mock')

    args = parser.parse_args()
    outdir = path.abspath(args.outdir)

    if args.config_file:
        config = Configuration(args.config_file)
    else:
        config = Configuration(path.join(path.dirname(outdir), 'pcigale.ini'))

    if len(sys.argv) == 1:
        parser.print_usage()
    else:
        if args.parser == 'chi2':
            chi2_action(config, outdir)
        elif args.parser == 'pdf':
            pdf_action(config, outdir)
        elif args.parser == 'sed':
            sed_action(config, args.type, args.nologo, outdir)
        elif args.parser == 'mock':
            mock_action(config, args.nologo, outdir)
