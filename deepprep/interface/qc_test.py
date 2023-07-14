# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from pathlib import Path
from qc_node import QCreport
from nipype import Node
from run import set_envrion
from config import settings as main_settings
def QCreport_test(subject_id, settings):
    qcreport_node = Node(QCreport(), f'QCreport_node')

    qcreport_dir = f'/mnt/ngshare/deepprep_test/qc_test/data'
    deepprep_home = settings.DEEPPREP_HOME
    # os.environ['SUBJECTS_DIR'] = subjects_dir

    qcreport_node.inputs.qcreport_dir = qcreport_dir
    qcreport_node.inputs.subject_id = subject_id
    qcreport_node.inputs.deepprep_home = deepprep_home
    qcreport_node.inputs.boilerplate_dir = Path(deepprep_home) / 'data' / 'logs'

    qcreport_node.run()

if __name__ == '__main__':
    set_envrion()

    subject_id = 'sub-MSC01'
    QCreport_test(subject_id, main_settings)