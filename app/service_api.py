import logging, sys
import pandas as pd
from flask import Flask, request, jsonify
from skops.io import load, get_untrusted_types
from marshmallow import Schema, fields, ValidationError


# Define valid input schema
class InputSchema(Schema):
  X1_transaction_date = fields.Float(required=True)
  X2_house_age = fields.Float(required=True)
  X3_distance_to_the_nearest_MRT_station = fields.Float(required=True)
  X4_number_of_convenience_stores = fields.Integer(required=True)
  X5_latitude = fields.Float(required=True)
  X6_longitude = fields.Float(required=True)


def create_app():
  # Set up logging
  file_handler = logging.FileHandler('app.log')
  file_handler.setLevel(logging.INFO)
  formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
  file_handler.setFormatter(formatter)
  
  # Initialize API
  app = Flask(__name__)
  if not app.logger.handlers:
    app.logger.addHandler(file_handler)
  app.logger.setLevel(logging.INFO)

  app.logger.info("Service API is running and logging is active.")

  # Load model at startup
  model_path = 'ml/model.skops'
  unknown_types = get_untrusted_types(file=model_path)
  try:
    model = load(model_path, trusted=unknown_types)
    app.logger.info(f"Successfully loaded model from {model_path}.")
  except Exception as e:
    app.logger.warning(f"Failed to load model from {model_path}: {e}")
    sys.exit(1) # stop running if model doesn't load correctly

  # Attach model to app
  app.model = model
  
  input_schema = InputSchema()


  """
  Run model inference on given input sample.
  Returns: Predicted price if input is valid, error otherwise.
  """
  @app.route('/predict', methods=['POST'])
  def predict():
    logging_info = f"{request.method} {request.path}"

    try:
      # Get JSON input
      json_input = request.get_json()
      app.logger.info(f"{logging_info} - Received input: {json_input}")

      # Validate input
      validated_input = input_schema.load(json_input)
      app.logger.info(f"{logging_info} - Input passed validation: {validated_input}.")

      # Convert to dataframe
      input_df = pd.DataFrame([validated_input])

      # Perform inference
      predicted_price = float(app.model.predict(input_df)[0]) # native type ensures correct JSON serialization
      app.logger.info(f"{logging_info} - Prediction result: {predicted_price}.")

      # Return prediction
      return jsonify({"predicted_price": predicted_price}), 200 # OK

    except ValidationError as err:
      # Throw error if validation fails
      app.logger.warning(f"{logging_info} - Input failed validation: {err.messages}")
      return jsonify({"error": err.messages}), 400 # bad request


  """
  Get model health.
  Returns: Static OK response.
  """
  @app.route('/health', methods=['GET'])
  def health():
    logging_info = f"{request.method} {request.path}"
    message = "Model is healthy"
    app.logger.info(f"{logging_info} - {message}")
    return message, 200 # OK

    
  return app


def main(host='127.0.0.1'):
  app = create_app()
  app.run(host=host, port=5000, debug=True)


if __name__ == "__main__":
  # Use localhost when running it locally, 0.0.0.0 for Docker
  host = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
  main(host)
