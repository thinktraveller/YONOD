#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=hyperparams
#SBATCH --output=%j_%A_%a.out
#SBATCH --mem=250G
#SBATCH --array=0-4
#SBATCH --cpus-per-task=2

cd $WRKDIR
module load mamba
source activate gecko
cd ATMOMACCS

case $SLURM_ARRAY_TASK_ID in
   0)  SEED=432 ;;
   1)  SEED=5  ;;
   2)  SEED=7543  ;;
   3)  SEED=12343  ;;
   4)  SEED=452 ;;
esac

# Record the start time
start_time=$(date +%s)

echo "start_time: $start_time"

srun python3 src/model/hyperparameter_gridsearch.py -ds 'Ferraz-Caetano' -t 'dvap' -s $SEED

# Record the end time
end_time=$(date +%s)

echo "end_time: $end_time"

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime
echo "Total runtime: $runtime seconds"