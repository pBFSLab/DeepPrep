from pkg_resources import resource_filename as pkgrf
import logging
from multiprocessing import Process, Manager
from pathlib import Path
from niworkflows.reports.core import Report as _Report
import os
# This patch is intended to permit fMRIPrep 20.2.0 LTS to use the YODA-style
# derivatives directory. Ideally, we will remove this in 20.3.x and use an
# updated niworkflows.


class Report(_Report):
    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        # Removed from here: Appending self.packagename to self.root and self.out_dir
        # In this version, pass reportlets_dir and out_dir with fmriprep in the path.

        if self.subject_id is not None:
            self.root = self.root / "sub-{}".format(self.subject_id)

        if "template_path" in settings:
            self.template_path = config.parent / settings["template_path"]

        self.index(settings["sections"])


def run_reports(
    out_dir,
    subject_id,
    run_uuid,
    config=None,
    reportlets_dir=None,
    packagename=None,
):

    return Report(
        out_dir,
        run_uuid,
        config=config,
        subject_id=subject_id,
        packagename=packagename,
        reportlets_dir=reportlets_dir,
    ).generate_report()


def generate_reports(
    subject_list, output_dir, run_uuid, config=None, work_dir=None, packagename=None
):
    """Execute run_reports on a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"
    report_errors = [
        run_reports(
            output_dir,
            subject_id,
            run_uuid,
            config=config,
            packagename=packagename,
            reportlets_dir=reportlets_dir,
        )
        for subject_id in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            "%s (%d)" % (subid, err)
            for subid, err in zip(subject_list, report_errors)
            if err
        )
        logger.error(
            "Preprocessing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno

# tmpdir = f'/home/lincong/workspace/DeepPrep/deepprep/data/test'
# cwd = os.getcwd()
# os.chdir(tmpdir)
#
# testdir = Path(f'/data/test')
#
# run_reports(testdir / 'out', 'MSC01', 'madeoutuuid', config=Path(f'/data/reports-spec-deepprep.yml'),
#             packagename='deepprep', reportlets_dir=testdir / 'work' / 'reportlets' / 'deepprep_test')

