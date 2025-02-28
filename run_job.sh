#!/bin/bash --login
#SBATCH -J DDIM_inference               # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=a100-galvani   # Which partition will run your job
#SBATCH --time=2-00:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:1               # (optional) Requesting type and number of GPUs
#SBATCH --mem=50G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=DDIM/logs/output/job-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=DDIM/logs/error/myjob-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=FAIL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=bela.umlauf@student.uni-tuebingen.de   # Email to which notifications will be sent

conda activate thesis

# Run our code
echo "-------- PYTHON OUTPUT ----------"


export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 -m DDIM.main configs/DDIM_config.yaml

echo "---------------------------------"

# Deactivate environment again
conda deactivate
