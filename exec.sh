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

# go to the directory that has app.py
cd /home/seif_elkerdany/projects/Attendance_System
export FLASK_APP=Frontend.app

source /home/seif_elkerdany/main_ML/bin/activate main_ML

# explicitly use the venv with flask
# VENV_PATH="/home/seif_elkerdany/main_ML/bin/activate main_ML"
python3 -m flask run


# ---- to run the file ---- 
# chmod +x exec.sh
# ./exec.sh