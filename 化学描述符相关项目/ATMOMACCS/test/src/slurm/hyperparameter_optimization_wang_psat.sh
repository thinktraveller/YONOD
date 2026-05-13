#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=hyperparams
#SBATCH --output=%j_%A_%a.out
#SBATCH --mem=60G
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2

cd $WRKDIR
module load mamba
source activate gecko
cd ATMOMACCS    

case $SLURM_ARRAY_TASK_ID in
   0)  SEED=12 ;;
   1)  SEED=325432435  ;;
   2)  SEED=326  ;;
   3)  SEED=436  ;;
   4)  SEED=2435 ;;
   5)  SEED=432 ;;
   6)  SEED=5  ;;
   7)  SEED=7543  ;;
   8)  SEED=12343  ;;
   9)  SEED=452 ;;
esac

# Record the start time
start_time=$(date +%s)

srun python3 src/model/hyperparameter_gridsearch.py -ds 'Wang' -t log_p_sat -s $SEED
srun python3 src/model/hyperparameter_gridsearch.py -ds 'Wang' -t log_kwg -s $SEED
srun python3 src/model/hyperparameter_gridsearch.py -ds 'Wang' -t log_kwiomg -s $SEED
# Record the end time
end_time=$(date +%s)

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime
echo "Total runtime: $runtime seconds"