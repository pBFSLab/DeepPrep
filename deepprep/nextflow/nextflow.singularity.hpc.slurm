#!/bin/bash

#SBATCH --job-name=multi  # 任务名称
#SBATCH --partition=cpu1,cpu2,fat,amd1,gpu1,gpu2  # 队列名称

#SBATCH --nodes=1  # 使用的节点数量（如果没有使用MPI，设置1）
#SBATCH --ntasks=1  # 每个节点执行的任务数量（如果没有使用srun执行命令，设置1）

#SBATCH --mem=4G  # 使用的RAM数量
#SBATCH --cpus-per-task=4  # 每个任务申请的CPU核心数量

#SBATCH --output=%x.%j.out  # 输出的信息日志
#SBATCH --error=%x.%j.err  # 输出的错误日志
#SBATCH --mail-type=end  # 运行结束后发送邮件
#SBATCH --mail-user=abcdef@g.com  # 收邮件的邮箱


nextflow run /lustre/grp/lhslab/sunzy/anning/workspace/DeepPrep/deepprep/nextflow/deepprep.nf \
-resume \
-c /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/nextflow.singularity.hpc.config \
--bids_dir /lustre/grp/lhslab/sunzy/BIDS/HNU \
--subjects_dir /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/HNU_v0.0.9ubuntu22.04H/Recon \
--bold_preprocess_path /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/HNU_v0.0.9ubuntu22.04H/BOLD \
--qc_result_path /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/HNU_v0.0.9ubuntu22.04H/QC \
-with-report /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/HNU_v0.0.9ubuntu22.04H/QC/report.html \
-with-timeline /lustre/grp/lhslab/sunzy/anning/DEEPPREP_WORKDIR/HNU_v0.0.9ubuntu22.04H/QC/timeline.html \
--bold_task_type rest
