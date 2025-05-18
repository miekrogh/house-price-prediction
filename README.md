# house-price-prediction
Deploy a house price prediction service (3-hour time limit)


## Setup

Create and activate virtual enviroment:

```bash
python -m venv venv
source venv/bin/activate # Windows: source venv\Scripts\activate
```

All required Python packages are listed in `requirements.txt`. To install the dependencies, run the following:

```bash
pip install -r requirements.txt
```


## Train the model

Run the following: 

```bash
python ml/model_training.py
```

This will train a linear regression model on the provided dataset and save it to memory as `ml/model.skops`.
