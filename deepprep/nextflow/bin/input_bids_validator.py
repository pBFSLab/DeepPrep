#! /usr/bin/env python3

import argparse

from pathlib import Path

from fmriprep.utils.bids import validate_input_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: validate input BIDS directories"
    )
    parser.add_argument("--bids_dir", required=True, help="directory of bids")
    parser.add_argument("--exec_env", required=True, help="docker or singularity")
    parser.add_argument("--participant_label", nargs='+', default=[], required=False)
    parser.add_argument("--skip_bids_validation", required=False, default=False)
    args = parser.parse_args()

    # Validate inputs
    skip_bids_validation = False if args.skip_bids_validation=='False' else True
    participant_label = None if args.participant_label=='None' else args.participant_label
    bids_dir = Path(args.bids_dir)
    if not skip_bids_validation:
        validate_input_dir(args.exec_env, bids_dir, participant_label)
