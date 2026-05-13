#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=krr_geckoq
#SBATCH --output=%j_%A_%a.out
#SBATCH --mem=140G
#SBATCH --array=0-9
#SBATCH --cpus-per-task=2

cd $WRKDIR
module load mamba
source activate gecko
cd ATMOMACCS_clean/ATMOMACCS

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

for v in {1..4}
do
   srun python3 src/main.py -v $v -d 'data/GeckoQ' -t 'dvap.txt' -ds 'GeckoQ' -s $SEED
done


srun python3 src/main.py -v 4 -d 'data/GeckoQ' -t 'dvap.txt' -ds 'GeckoQ' -s $SEED -b 'ATMOMACCS_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/GeckoQ' -t 'dvap.txt' -ds 'GeckoQ' -s $SEED -b 'ATMO_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/GeckoQ' -t 'dvap.txt' -ds 'GeckoQ' -s $SEED -b 'ATMO'
srun python3 src/main.py -v 3 -d 'data/GeckoQ' -t 'dvap.txt' -ds 'GeckoQ' -s $SEED -b 'ATMOMACCS_ALT'
srun python3 src/main.py -v 0 -d 'data/GeckoQ' -t 'dvap.txt' -ds 'GeckoQ' -s $SEED -b 'MACCS'


# Record the end time
end_time=$(date +%s)

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime
echo "Total runtime: $runtime seconds"