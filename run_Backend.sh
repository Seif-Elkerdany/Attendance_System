# Exit immediately if a command fails.
set -e

# Change to the directory of the script 
cd "$(dirname "$0")"

echo "Project root: $(pwd)"

# Ensure necessary __init__.py files exist
required_packages=("modeling" "modeling/model" "src")
for pkg in "${required_packages[@]}"; do
    if [ -d "$pkg" ]; then
        if [ ! -f "$pkg/__init__.py" ]; then
            echo "Creating $pkg/__init__.py"
            touch "$pkg/__init__.py"
        fi
    else
        echo "Warning: Directory '$pkg' not found. Please check your project structure."
    fi
done

# Set PYTHONPATH to the project root
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# Activate your Python environment
source /home/seif_elkerdany/main_ML/bin/activate main_ML

# Run the Backend module
echo "Starting the Attendance System..."
python -m src.Backend

# To run:
# chmod +x run_Backend.sh
# ./run_Backend.sh
