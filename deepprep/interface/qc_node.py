# python3
# -*- coding: utf-8 -*-
# -------------------------------
# @Author : Ning An        @Email : NingAnMe <ninganme0317@gmail.com>
# @Author : Cong Lin       @Email : lincong <lincong8722@gmail.com>
# @Author : Youjia Zhang   @Email : youjia <ireneyou33@gmail.com>
# @Author : Zhenyu Sun     @Email : Kid-sunzhenyu <sun25939789@gmail.com>

from pathlib import Path
from nipype.interfaces.base import BaseInterfaceInputSpec, BaseInterface, File, TraitedSpec, Directory, Str
from deepprep.interface.run import run_cmd_with_timing, multipool
from reports.core import run_reports
import os
from uuid import uuid4
from time import strftime

class QCreportInputSpec(BaseInterfaceInputSpec):
    qcreport_dir = Directory(exists=True, desc='QCreport dir path', mandatory=True)
    boilerplate_dir = Directory(exists=True, desc='citation dir path', mandatory=True)
    subject_id = Str(desc='subject id', mandatory=True)
    deepprep_home = Directory(exists=True, desc='DEEPPREP_HOME', mandatory=True)


class QCreportOutputSpec(TraitedSpec):
    QCreport_file = File(exists=True, desc='QCreport')


class QCreport(BaseInterface):
    input_spec = QCreportInputSpec
    output_spec = QCreportOutputSpec

    def __init__(self):
        super(QCreport, self).__init__()

    def _run_interface(self, runtime):
        out_dir = Path(self.inputs.qcreport_dir) / 'output'
        report_dir = out_dir / self.inputs.subject_id
        log_dir = report_dir / 'logs'
        reports_spec = Path(self.inputs.deepprep_home) / 'data' / 'reports-spec-deepprep.yml'
        run_uuid = f"{strftime('%Y%m%d-%H%M%S')}_{uuid4()}"
        if not log_dir.exists():
            cmd = f'cp -r {self.inputs.boilerplate_dir} {log_dir}'
            os.system(cmd)
        run_reports(report_dir, self.inputs.subject_id, run_uuid,
                    config=reports_spec, packagename='deepprep',
                    reportlets_dir=out_dir)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['QCreport_file'] = Path(self.inputs.qcreport_dir) / 'output' / self.inputs.subject_id / f'{self.inputs.subject_id}.html'
        return outputs

    def create_sub_node(self, settings):
        return []
