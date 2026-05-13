#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --job-name=krr_all
#SBATCH --output=%j_%A_%a.out
#SBATCH --mem=20G
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1

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
   srun python3 src/main.py -v $v -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED
   srun python3 src/main.py -v $v -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED
   srun python3 src/main.py -v $v -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED
   srun python3 src/main.py -v $v -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED
   srun python3 src/main.py -v $v -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED
      
   #srun python3 src/main.py -v $v -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'True'
   #srun python3 src/main.py -v $v -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'True'
   #srun python3 src/main.py -v $v -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'True'
   #srun python3 src/main.py -v $v -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'True'
   #srun python3 src/main.py -v $v -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'True'
done


srun python3 src/main.py -v 4 -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'ATMOMACCS_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'ATMOMACCS_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'ATMOMACCS_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'ATMOMACCS_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'ATMOMACCS_DECIMAL'

srun python3 src/main.py -v 4 -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'ATMO_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'ATMO_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'ATMO_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'ATMO_DECIMAL'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'ATMO_DECIMAL'


srun python3 src/main.py -v 4 -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'ATMO'
srun python3 src/main.py -v 4 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'ATMO'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'ATMO'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'ATMO'
srun python3 src/main.py -v 4 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'ATMO'

srun python3 src/main.py -v 3 -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'ATMOMACCS_ALT'
srun python3 src/main.py -v 3 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'ATMOMACCS_ALT'
srun python3 src/main.py -v 3 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'ATMOMACCS_ALT'
srun python3 src/main.py -v 3 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'ATMOMACCS_ALT'
srun python3 src/main.py -v 3 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'ATMOMACCS_ALT'

srun python3 src/main.py -v 0 -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'TopFP_tg'
srun python3 src/main.py -v 0 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'TopFP_dvap'
srun python3 src/main.py -v 0 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'TopFP_log_p_sat'
srun python3 src/main.py -v 0 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'TopFP_log_kwiomg'
srun python3 src/main.py -v 0 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'TopFP_log_kwg'

srun python3 src/main.py -v 0 -d 'data/Li' -t 'tg.txt' -ds 'Li' -s $SEED -b 'MACCS'
srun python3 src/main.py -v 0 -d 'data/Ferraz-Caetano' -t 'dvap.txt' -ds 'Ferraz-Caetano' -s $SEED -b 'MACCS'
srun python3 src/main.py -v 0 -d 'data/Wang' -t 'log_p_sat.txt' -ds 'Wang' -s $SEED -b 'MACCS'
srun python3 src/main.py -v 0 -d 'data/Wang' -t 'log_kwiomg.txt' -ds 'Wang' -s $SEED -b 'MACCS'
srun python3 src/main.py -v 0 -d 'data/Wang' -t 'log_kwg.txt' -ds 'Wang' -s $SEED -b 'MACCS'

# Record the end time
end_time=$(date +%s)

# Calculate the total runtime
runtime=$((end_time - start_time))

# Print the total runtime
echo "Total runtime: $runtime seconds"