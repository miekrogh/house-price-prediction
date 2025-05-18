import pytest, json
from app.service_api import create_app
from sklearn.utils.validation import check_is_fitted


@pytest.fixture
def app():
  app = create_app()
  app.config.update({"TESTING": True})
  return app

@pytest.fixture
def client(app):
  return app.test_client()


"""
The model is loaded correctly when starting up the app.
"""
def test_model_loads_correctly(app):
  model = app.model
  assert model is not None
  assert hasattr(model, 'predict')
  check_is_fitted(model)


"""
The /predict endpoint returns predicted price and status code 200 (OK) when given valid JSON input.
"""
def test_predict_returns_predicted_price_on_valid_input(client):
  valid_input = {
      "X1_transaction_date": 2013.333,
      "X2_house_age": 6.3,
      "X3_distance_to_the_nearest_MRT_station": 90.45606,
      "X4_number_of_convenience_stores": 9,
      "X5_latitude": 24.97433,
      "X6_longitude": 121.5431
  }

  response = client.post('/predict', data=json.dumps(valid_input), content_type='application/json')
  assert response.status_code == 200
  
  data = response.get_json()
  assert 'predicted_price' in data
  assert isinstance(data['predicted_price'], float)


"""
The /predict endpoint returns status code 400 (bad request) when given invalid input.
"""
def test_predict_returns_status_code_400_on_invalid_input(client):
  invalid_input = {
    'X1_transaction_date': 2013.333,
    'X2_house_age': 6.3,
    'X3_distance_to_the_nearest_MRT_station': 90.45606
    # missing three features
  }

  response = client.post('/predict', data=json.dumps(invalid_input), content_type='application/json')
  assert response.status_code == 400

  data = response.get_json()
  assert 'error' in data


"""
The /health endpoint always returns status code 200 (OK).
"""
def test_health_returns_status_code_200(client):
  response = client.get('/health')
  assert response.status_code == 200
  assert response.get_data(as_text=True) == "Model is healthy" # decode byte response
