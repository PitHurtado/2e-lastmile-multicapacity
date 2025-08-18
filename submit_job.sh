# Initialize Modules
source /etc/profile

# Load Anaconda and Gurobi Modules
module load anaconda/2023a
module load gurobi/gurobi-1000

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

# run the script
python -u ./run_evaluation.py $LLSUB_RANK $LLSUB_SIZE