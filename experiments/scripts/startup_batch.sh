#!/bin/bash
# (See https://arc-ts.umich.edu/greatlakes/user-guide/ for command details)

# Set up batch job settings
#SBATCH --job-name=../outputs/MindLSTMBatch_estop
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=100GB
#SBATCH --account=eecs692w23_class
#SBATCH --time=10:00:00
#SBATCH --output=%x-output.log

MODEL="LSTM"
DLG="Yes"
POV="First"
VID="Yes"
STYLE="attn"

module load python/3.10.4
module load cuda
cd ..
source venv/bin/activate
for EXP_NUM in $(seq 3 8);
do
    SAVE_PATH="/scratch/eecs692w23_class_root/eecs692w23_class/anrao/clip/pov_${POV}_${MODEL}_${EXP_NUM}.torch"
    python3 sandbox.py --clip --train --pov ${POV} --use_dialogue ${DLG} --plans Yes --seq_model ${MODEL} --experiment ${EXP_NUM} --save_path $SAVE_PATH --seed Fixed > outputs/saved_results/${STYLE}/exp_${EXP_NUM}/out_${MODEL}_CLIP_${VID}_${EXP_NUM}_lastd.txt
done
