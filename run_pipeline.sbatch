#!/bin/bash
#
# run_reddit_ideology_array.sbatch
#
#SBATCH --job-name=reddit_ideology
#SBATCH --output=logs/reddit_ideology_%A_%a.out
#SBATCH --error=logs/reddit_ideology_%A_%a.err
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --array=1-3

# === Your email here ===
MAILTO=isaacharlem@uchicago.edu

# === Config array ===
CONFIGS=(
  "my_config_M.yaml"
  "my_config_Q.yaml"
  "my_config_Y.yaml"
)
CFG=${CONFIGS[$(( SLURM_ARRAY_TASK_ID - 1 ))]}

# === Conda setup & cwd ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate red_id
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# === Locate mail binaries ===
echo "Which mail: $(which mail 2>/dev/null || echo 'not found')" 
echo "Which sendmail: $(which sendmail 2>/dev/null || echo 'not found')"

# === START notification ===
START_MSG="SLURM job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID starting with config $CFG on $(hostname) at $(date)"
# try mail, fallback to sendmail, else warn
echo "$START_MSG" \
  | /bin/mail -s "SLURM job $SLURM_JOB_ID START" $MAILTO \
  || echo "WARNING: /bin/mail failed" >&2
echo "$START_MSG" \
  | /usr/sbin/sendmail -t "$MAILTO" \
  || echo "WARNING: /usr/sbin/sendmail failed" >&2

# === Run pipeline ===
reddit-ideology --config "$CFG" run

# === FINISH notification ===
FIN_MSG="SLURM job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID finished on $(hostname) at $(date)"
echo "$FIN_MSG" \
  | /bin/mail -s "SLURM job $SLURM_JOB_ID FINISH" $MAILTO \
  || echo "WARNING: /bin/mail failed" >&2
echo "$FIN_MSG" \
  | /usr/sbin/sendmail -t "$MAILTO" \
  || echo "WARNING: /usr/sbin/sendmail failed" >&2
