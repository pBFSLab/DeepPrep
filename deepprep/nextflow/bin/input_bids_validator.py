import argparse

from fmriprep.utils.bids import validate_input_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="DeepPrep: validate input BIDS directories"
    )
    parser.add_argument("--bids_dir", required=True, help="directory of bids")
    parser.add_argument("--exec_env", required=True, help="directory of results")
    parser.add_argument("--participant_label", required=True, help="directory of Recon results")
    parser.add_argument("--skip_bids_validation", type=str, nargs='+', required=True, help="type of bold space outputs")
    args = parser.parse_args()

    # Validate inputs
    if not args.skip_bids_validation:
        validate_input_dir(args.exec_env, args.bids_dir, args.participant_label)