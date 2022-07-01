import os
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bd', required=True, help='directory of bids type')
    parser.add_argument('--fsd', default=os.environ.get('FREESURFER_HOME'),
                        help='Output directory $FREESURFER_HOME (pass via environment or here)')
    parser.add_argument('--python', default='python3',
                        help='which python version to use')

    args = parser.parse_args()
    args_dict = vars(args)

    if args.fsd is None:
        args_dict['fsd'] = '/usr/local/freesurfer'

    return argparse.Namespace(**args_dict)


if __name__ == '__main__':
    args = parse_args()

    abs_dir = os.path.dirname(os.path.abspath(__file__))
    structure_py = os.path.join(abs_dir, 'structure_preprocess_fs720.py')
    bold_py = os.path.join(abs_dir, 'bold_preprocess.py')

    s_time = time.time()
    os.system(f'{args.python} {structure_py} --bd {args.bd} --fsd {args.fsd} --python {args.python}')
    t1_time = time.time() - s_time
    os.system(f'{args.python} {bold_py} --bd {args.bd} --fsd {args.fsd}')
    bold_time = time.time() - s_time - t1_time
    all_time = time.time() - s_time
    print(f"T1 : {t1_time}")
    print(f"BOLD : {bold_time}")
    print(f"ALL : {all_time}")

    print('Done!')
