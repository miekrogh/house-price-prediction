
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
  # Initialize API
  app = Flask(__name__)

  # Load model at startup
  model_path = 'ml/model.skops'
  unknown_types = get_untrusted_types(file=model_path)
  model = load(model_path, trusted=unknown_types)

  # Attach model to app
  app.model = model

  input_schema = InputSchema()


  @app.route('/predict', methods=['POST'])
  def predict():
    try:
      # Get JSON input
      json_input = request.get_json()

      # Validate input
      validated_input = input_schema.load(json_input)

      # Convert to dataframe
      input_df = pd.DataFrame([validated_input])

      # Perform inference
      predicted_price = float(app.model.predict(input_df)[0]) # native type ensures correct JSON serialization

      # Return prediction
      return jsonify({"predicted_price": predicted_price}), 200 # OK

    except ValidationError as err:
      # Throw error if validation fails
      return jsonify({"error": err.messages}), 400 # bad request


def main():
  app = create_app()
  app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == "__main__":
  main()