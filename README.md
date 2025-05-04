# PTC-Praxis-Capstone-project
This capstone project aims to implement and compare a portfolio design method based on hierarchical clustering (Hierarchical Risk Parity) against the traditional Markowitz mean-variance optimization method.

## Local Setup

### Create conda environment

In your VSCode workspace/Explorer section, right-click and choose the option `Open in Integrated Terminal`.

In the terminal, execute the following commands:
```
conda create --name ptc-praxis-capstone python=3.12
conda activate ptc-praxis-capstone
pip install -r requirements.txt
conda deactivate
```

### Run the app using:
```
streamlit run data_analysis.py
```