#!/bin/bash
#SBATCH --job-name=C2C
#SBATCH --time=05:59:59
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4
#SBATCH --output=C2C-%j.out
#SBATCH --cpus-per-task=40
#SBATCH --mem=164gb
#SBATCH -p a100


### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=12340
# export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
echo "SLURM_NTASKS="${SLURM_NTASKS}
echo "SLURM_PROCID="${SLURM_PROCID}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load cuda/11.8
source activate C2C

# accelerate config
echo "Starting accelerate..."
srun python3 train.py \
    --config_file cfg/segformer.json\
    --checkpoint_dir runs/segformer_exp0\
    --epochs 100\
    --experiment_name training_exp0
# srun python3 train.py --config_file cfg/segformer.json --checkpoint_dir runs/segformer_exp0 --epochs 100 --experiment_name training_exp0