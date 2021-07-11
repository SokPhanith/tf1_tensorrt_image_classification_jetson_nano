#!/usr/bin/env python3
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

"""
convert_to_uff.py

Main script for doing uff conversions from
different frameworks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
# Make sure we import the correct UFF
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import argparse
import uff


def _replace_ext(path, ext):
    return os.path.splitext(path)[0] + ext

def process_cmdline_args():
    """
    Helper function for processing commandline arguments
    """
    parser = argparse.ArgumentParser(description="""Converts TensorFlow models to Unified Framework Format (UFF).""")

    parser.add_argument(
        "input_file",
        help="""path to input model (protobuf file of frozen GraphDef)""")

    parser.add_argument(
        '-l', '--list-nodes', action='store_true',
        help="""show list of nodes contained in input file""")

    parser.add_argument(
        '-t', '--text', action='store_true',
        help="""write a text version of the output in addition to the
        binary""")

    parser.add_argument(
        '--write_preprocessed', action='store_true',
        help="""write the preprocessed protobuf in addition to the
        binary""")

    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help="""disable log messages""")

    parser.add_argument(
        '-d', '--debug', action='store_true',
        help="""Enables debug mode to provide helpful debugging output""")

    parser.add_argument(
        "-o", "--output",
        help="""name of output uff file""")

    parser.add_argument(
        "-O", "--output-node", default=[], action='append',
        help="""name of output nodes of the model""")

    parser.add_argument(
        '-I', '--input-node', default=[], action='append',
        help="""name of a node to replace with an input to the model.
        Must be specified as: "name,new_name,dtype,dim1,dim2,..."
        """)

    parser.add_argument(
        "-p", "--preprocessor",
        help="""the preprocessing file to run before handling the graph. This file must define a `preprocess` function that accepts a GraphSurgeon DynamicGraph as it's input. All transformations should happen in place on the graph, as return values are discarded""")

    args, _ = parser.parse_known_args()
    args.output = _replace_ext((args.output if args.output else args.input_file), ".uff")
    return args, _

def main():
    args, _ = process_cmdline_args()
    if not args.quiet:
        print("Loading", args.input_file)
    uff.from_tensorflow_frozen_model(
        args.input_file,
        output_nodes=args.output_node,
        preprocessor=args.preprocessor,
        input_node=args.input_node,
        quiet=args.quiet,
        text=args.text,
        list_nodes=args.list_nodes,
        output_filename=args.output,
        write_preprocessed=args.write_preprocessed,
        debug_mode=args.debug
    )

if __name__ == '__main__':
    main()
