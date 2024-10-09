conda create --prefix ./.cenv python=3.11
conda activate ./.cenv

pip install --upgrade python-dotenv wandb

pip install --upgrade optimum accelerate
pip install --upgrade autoawq
