source ~/.bashrc
conda activate dreamer

echo "Initializing Dreamer World Model training..."

# Change to project directory
cd /ocean/projects/cis250269p/ywu18/Forewarn

python scripts/train_wm_real_data.py --config_path wm_cup_vla_config.yaml
