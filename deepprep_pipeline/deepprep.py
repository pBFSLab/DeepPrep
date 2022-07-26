import os
import argparse
import time


def link_bd_to_od(b_dir, o_dir):
    filenames = os.listdir(b_dir)
    for filename in filenames:
        if filename == 'derivatives':
            continue
        f_src = os.path.join(b_dir, filename)
        f_dst = os.path.join(o_dir, filename)
        if not os.path.exists(f_dst):
            os.symlink(f_src, f_dst)
    derivatives_path = os.path.join(o_dir, 'derivatives')
    if not os.path.exists(derivatives_path):
        os.mkdir(derivatives_path)


def parse_args():
    def _drop_sub(value):
        return value[4:] if value.startswith("sub-") else value

    parser = argparse.ArgumentParser()
    parser.add_argument('--bd', required=True, help='directory of bids type')
    parser.add_argument('--od', default=None, help='output directory')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--only-bold', default='off', help='only process bold')
    parser.add_argument('--only-anat', default='off', help='only process anat')
    parser.add_argument('--python', default='python3',
                        help='which python version to use')

    # anat preprocess
    parser.add_argument('--respective', default='off',
                        help='if on, while processing T1w file respectively')
    parser.add_argument('--rewrite', default='on',
                        help='set off, while not preprocess if subject recon path exist')

    # bold preprocess
    parser.add_argument("-t", "--task", action='store', nargs='+',
                        help='a space delimited list of tasks identifiers or a single task')
    parser.add_argument("-s", "--subject", action="store", nargs="+", type=_drop_sub,
                        help="a space delimited list of subject identifiers or a single "
                             "identifier (the sub- prefix can be removed)")

    args = parser.parse_args()
    args_dict = vars(args)

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'
    args_dict['only_bold'] = True if args_dict['only_bold'] == 'on' else False
    args_dict['only_anat'] = True if args_dict['only_anat'] == 'on' else False

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    args = parse_args()

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    structure_py = os.path.join(abs_dir, 'structure_preprocess_fs720.py')
    bold_py = os.path.join(abs_dir, 'bold_preprocess.py')

    if args.od == args.bd or args.od is None:
        bd = args.bd
    else:
        link_bd_to_od(args.bd, args.od)
        bd = args.od

    assert not (args.only_bold and args.only_anat), 'only-bold and only-anat can`t be on in sametime'

    anat = False if args.only_bold else True
    bold = False if args.only_anat else True

    s_time = time.time()
    if anat:
        os.system(f'{args.python} {structure_py} --bd {bd} --fsd {args.fsd} --python {args.python} '
                  f'--respective {args.respective} --rewrite {args.rewrite}')
    t1_time = time.time() - s_time
    if bold:
        cmd = f'{args.python} {bold_py} --bd {bd} --fsd {args.fsd}'
        if args.task is not None:
            cmd += ' -t '
            cmd += ' '.join(args.task)
        if args.subject is not None:
            cmd += ' -s '
            cmd += ' '.join(args.subject)
        os.system(cmd)
    bold_time = time.time() - s_time - t1_time
    all_time = time.time() - s_time
    print(f"T1 : {t1_time}")
    print(f"BOLD : {bold_time}")
    print(f"ALL : {all_time}")
    print('Done!')
