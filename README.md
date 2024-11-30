# Battlesnake Robbery

## How to run
### Prerequisites:
1. Create a venv (optional)
`python -m venv ./.venv`

2. Install Requirements
`pip install -r requirements.txt`

### For visualizations:
1. Edit `visualize.py` for the correct visualizations and settings
2. run `python visualize.py`

### For training:
1. Edit `train.py` with the correct model
2. run `python train.py`

## For hosting Battlesnake server
1. Ensure the existence of `model_pipeline.pkl` (this should be created after running `train.py`)
2. run `python server.py`