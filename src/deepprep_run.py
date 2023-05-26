import os
import argparse
import sys
import time
from pathlib import Path
from functools import partial


def _path_exists(path, parser):
    """Ensure a given path exists."""
    if path is None or not Path(path).exists():
        raise parser.error(f"Path does not exist: <{path}>.")
    return Path(path).absolute()


def _is_file(path, parser):
    """Ensure a given path exists, and it is a file."""
    path = _path_exists(path, parser)
    if not path.is_file():
        raise parser.error(f"Path should point to a file (or symlink of file): <{path}>.")
    return path


def _min_one(value, parser):
    """Ensure an argument is not lower than 1."""
    value = int(value)
    if value < 1:
        raise parser.error("Argument can't be less than one.")
    return value


def _to_gb(value):
    scale = {"G": 1, "T": 10 ** 3, "M": 1e-3, "K": 1e-6, "B": 1e-9}
    digits = "".join([c for c in value if c.isdigit()])
    units = value[len(digits):] or "M"
    return int(digits) * scale[units[0]]


def _drop_sub(value):
    return value[4:] if value.startswith("sub-") else value


def _filter_pybids_none_any(dct):
    import bids
    return {
        k: bids.layout.Query.NONE
        if v is None
        else (bids.layout.Query.ANY if v == "*" else v)
        for k, v in dct.items()
    }


def _bids_filter(value, parser):
    from json import loads, JSONDecodeError

    if value:
        if Path(value).exists():
            try:
                return loads(Path(value).read_text(), object_hook=_filter_pybids_none_any)
            except JSONDecodeError:
                raise parser.error(f"JSON syntax error in: <{value}>.")
        else:
            raise parser.error(f"Path does not exist: <{value}>.")


def _slice_time_ref(value, parser):
    if value == "start":
        value = 0
    elif value == "middle":
        value = 0.5
    try:
        value = float(value)
    except ValueError:
        raise parser.error("Slice time reference must be number, 'start', or 'middle'. "
                           f"Received {value}.")
    if not 0 <= value <= 1:
        raise parser.error(f"Slice time reference must be in range 0-1. Received {value}.")
    return value


def link_bd_to_od(b_dir, o_dir):
    filenames = os.listdir(b_dir)
    for filename in filenames:
        if filename == "derivatives":
            continue
        f_src = os.path.join(b_dir, filename)
        f_dst = os.path.join(o_dir, filename)
        if not os.path.exists(f_dst):
            os.symlink(f_src, f_dst)
    derivatives_path = os.path.join(o_dir, "derivatives")
    if not os.path.exists(derivatives_path):
        os.mkdir(derivatives_path)


def parse_args():

    parser = argparse.ArgumentParser(
        description="DeepPrep: sMRI and fMRI PreProcessing workflows"
    )
    PathExists = partial(_path_exists, parser=parser)
    IsFile = partial(_is_file, parser=parser)
    PositiveInt = partial(_min_one, parser=parser)
    BIDSFilter = partial(_bids_filter, parser=parser)
    SliceTimeRef = partial(_slice_time_ref, parser=parser)

    parser.add_argument("bids_dir", help="directory of bids type")  # position parameter
    parser.add_argument("output_dir", help="output directory")  # position parameter
    parser.add_argument("--only-bold", action="store_true", default=False, help="only process bold")
    parser.add_argument("--only-anat", action="store_true", default=False, help="only process anat")

    # External Dependencies path
    parser.add_argument("--freesurfer_home", default=os.environ.get("FREESURFER_HOME"),
                        type=PathExists,
                        help="Output directory $FREESURFER_HOME (pass via environment or here)")

    parser.add_argument("--python", default="python3",
                        help="which python interpreter to use")

    # bids filter
    bids_g = parser.add_argument_group("Options for bids filter")
    bids_g.add_argument("--subject-id", action="store", nargs="+", type=_drop_sub,
                        help="a space delimited list of subject identifiers or a single "
                             "identifier (the sub- prefix can be removed)")
    # func_g.add_argument('--session-id', action='store', default='single_session',
    #                     help='select a specific session to be processed')
    # func_g.add_argument('--run-id', action='store', default='single_run',
    #                     help='select a specific run to be processed')

    # anat preprocess
    anat_g = parser.add_argument_group("Options for sMRI preprocess")
    anat_g.add_argument("--respective", action="store_true", default=False,
                        help="if True, while processing T1w file respectively")
    anat_g.add_argument("--rewrite", action="store_true", default=False,
                        help="if True, while preprocess even though subject recon path exist")

    # func preprocess
    func_g = parser.add_argument_group("Options for fMRI preprocess")
    func_g.add_argument("-t", "--task-id", action="store", nargs="+",
                        help="a space delimited list of tasks identifiers or a single task")
    func_g.add_argument("--anat-recon", action="store", type=PathExists,
                        help="while use a existed anat recon result for func preprocess")
    _args = parser.parse_args()

    return _args


if __name__ == "__main__":
    python_interpret = sys.executable
    print(python_interpret)
    args = parse_args()

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    structure_py = os.path.join(abs_dir, "structure_preprocess_fs720.py")
    bold_py = os.path.join(abs_dir, "bold_preprocess.py")

    if args.od == args.bids_dir or args.output_dir is None:
        bids_dir = args.bids_dir
    else:
        link_bd_to_od(args.bids_dir, args.output_dir)
        bids_dir = args.output_dir

    assert not (args.only_bold and args.only_anat), "only-bold and only-anat can`t be on in sametime"

    anat = False if args.only_bold else True
    bold = False if args.only_anat else True

    s_time = time.time()
    if anat:
        os.system(f"{args.python} {structure_py} --bd {bids_dir} --fsd {args.fsd} --python {args.python} "
                  f"--respective {args.respective} --rewrite {args.rewrite}")
    t1_time = time.time() - s_time
    if bold:
        cmd = f"{args.python} {bold_py} --bd {bids_dir} --fsd {args.fsd}"
        if args.task is not None:
            cmd += " -t "
            cmd += " ".join(args.task)
        if args.subject is not None:
            cmd += " -s "
            cmd += " ".join(args.subject)
        os.system(cmd)
    bold_time = time.time() - s_time - t1_time
    all_time = time.time() - s_time
    print(f"T1 : {t1_time}")
    print(f"BOLD : {bold_time}")
    print(f"ALL : {all_time}")
    print("Done!")
