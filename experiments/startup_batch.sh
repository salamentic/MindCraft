#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=outputs/EECS692MindCraft
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --mem-per-gpu=86GB
#SBATCH --account=eecs692w23_class
#SBATCH --time=16:00:00
#SBATCH --output=%x-output.log

MODEL="Transformer"
DLG="Yes"
POV="First"
VID="Yes"

module load python/3.10.4
module load cuda
source venv/bin/activate
for EXP_NUM in $(seq 3 8);
do
    SAVE_PATH="/scratch/eecs692w23_class_root/eecs692w23_class/anrao/out_with_plan/dialogue_${DLG}_pov_${POV}_Transformer_${EXP_NUM}.torch"
    python3 sandbox.py --clip --train --pov ${POV} --use_dialogue ${DLG} --plans Yes --seq_model ${MODEL} --experiment ${EXP_NUM} --save_path $SAVE_PATH --seed Fixed > outputs/saved_results/out_${MODEL}_CLIP_${VID}_${EXP_NUM}.txt
done
