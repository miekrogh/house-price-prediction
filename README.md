## Solution

I have implemented a simple ML pipeline using a linear regression model to predict house prices. 
While the data does not appear to be strictly linearly separable, I chose linear regression for its simplicity given the time constraints.
Minimal feature engineering has been performed: I dropped the 'No' column and normalized the features (zero mean, unit variance).
The model was trained using scikit-learn and serialized with Skops for safe loading. 
I implemented a simple API using Flask to expose the model for predictions, and I used Marshmallow to validate incoming JSON payloads.
I implemented basic logging to track inputs and system activity.
I used pytest for testing the API and model loading.
Finally, I used Docker for containerizing the application and GitHub Actions to automate testing.


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


## Run and use the API

Run the following to start the Flask app:

```bash
python app/service_api.py
```

In another terminal, run the following to send a request to the /predict endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"X1_transaction_date": 2013.333, "X2_house_age": 6.3, "X3_distance_to_the_nearest_MRT_station": 90.45606, "X4_number_of_convenience_stores": 9, "X5_latitude": 24.97433, "X6_longitude": 121.5431}'
```

This returns a response in JSON format with the predicted house price, e.g., `{ "predicted_price": 53.6277324116802 }`, and the request and response are both logged in the Flask app terminal and in the `app.log` file in the root folder.

Run the following to send a request to the /health endpoint:

```bash
curl -X GET http://127.0.0.1:5000/health
```

This returns the string `Model is healthy` in the terminal, and the request and response are both logged in the Flask app terminal and in the `app.log` file in the root folder.


## Run tests

Run the following:

```bash
pytest
```

This runs the tests in `tests/test_app.py`. The requests and responses are logged to the `app.log` file in the root folder.


## Run API through Docker container

Run the following to build a Docker image:

```bash
docker build -t house-price-api . # or whichever image name you prefer
```

Run the following to start the Flask app in the container:

```bash
docker run -p 5000:5000 house-price-api
```

Once the Docker container is running, you can interact with the API using the same `curl` commands described above.

To stop the container and remove unused resources, run the following:

```bash
docker stop house-price-api
docker container prune
```
