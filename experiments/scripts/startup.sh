#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=../outputs/EECS692MindCraftSingle
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100GB
#SBATCH --account=eecs692w23_class
#SBATCH --time=1:20:00
#SBATCH --output=%x-output.log

MODEL="X-Transformer"
DLG="Yes"
POV="First"
VID="Yes"
EXP_NUM=6

module load python/3.10.4
module load cuda
cd ..
source venv/bin/activate
SAVE_PATH="/scratch/eecs692w23_class_root/eecs692w23_class/anrao/clip/pov_${POV}_${MODEL}_${EXP_NUM}.torch"
python3 sandbox.py --clip --pov ${POV} --use_dialogue ${DLG} --plans Yes --seq_model ${MODEL} --experiment ${EXP_NUM} --save_path $SAVE_PATH --seed Fixed > outputs/saved_results/out_${MODEL}_${CLIP}_${VID}_${EXP_NUM}.txt
#python3 sandbox.py --clip --train --pov ${POV} --use_dialogue ${DLG} --plans Yes --seq_model ${MODEL} --experiment ${EXP_NUM} --save_path $SAVE_PATH --seed Fixed > outputs/saved_results/out_${MODEL}_${CLIP}_${VID}_${EXP_NUM}.txt
