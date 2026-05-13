#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --job-name=shap_noQ
#SBATCH --output=%j_%A_%a.out
#SBATCH --mem=200G
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1

cd $WRKDIR
module load mamba
source activate gecko
cd ATMOMACCS_clean/ATMOMACCS

# 12 2435, [432, 5, 7543, 12343, 452, 325432435, 326, 436]
# Record the start time
start_time=$(date +%s)

# Get the current task ID from the job array
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

# Run the main.py script with the current value of -v
srun python3 src/model/shap_vals.py -d ATMOMACCS_DECIMAL_v4 -t log_p_sat -f data/Wang -s $SEED
srun python3 src/model/shap_vals.py -d ATMOMACCS_DECIMAL_v4 -t log_kwg -f data/Wang -s $SEED
srun python3 src/model/shap_vals.py -d ATMOMACCS_DECIMAL_v4 -t log_kwiomg -f data/Wang -s $SEED
srun python3 src/model/shap_vals.py -d ATMOMACCS_DECIMAL_v4 -t dvap -f data/Ferraz-Caetano -s $SEED
srun python3 src/model/shap_vals.py -d ATMOMACCS_DECIMAL_v4 -t tg -f data/Li -s $SEED
# Record the end time
end_time=$(date +%s)

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime
echo "Total runtime: $runtime seconds"