#!/bin/bash

#SBATCH --job-name=multi  # 任务名称
#SBATCH --partition=cpu1,cpu2,fat,amd1,hygon,sugon  # 队列名称

#SBATCH --nodes=1  # 使用的节点数量（如果没有使用MPI，设置1）
#SBATCH --ntasks=1  # 每个节点执行的任务数量（如果没有使用srun执行命令，设置1）

#SBATCH --mem=2G  # 使用的RAM数量
#SBATCH --cpus-per-task=2  # 每个任务申请的CPU核心数量

#SBATCH --output=%x.%j.out  # 输出的信息日志
#SBATCH --error=%x.%j.err  # 输出的错误日志
#SBATCH --mail-type=end  # 运行结束后发送邮件
#SBATCH --mail-user=ninganme@qq.com  # 收邮件的邮箱

nextflow run /lustre/grp/lhslab/sunzy/anning/DeepPrep/deepprep/nextflow/deepprep_hpc/deepprep.nf -resume -with-dag flowchart.png -with-timeline timeline.html -with-report report.html
