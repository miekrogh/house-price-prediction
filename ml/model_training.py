import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from skops.io import dump


def main():
  # Load and preprocess data.
  df = pd.read_csv('data/real_estate.csv')
  df.columns = [c.strip().replace(" ", "_") for c in df.columns] # replace spaces, necessary for input validation
  df = df.drop(columns=['No']) # identifiers carry no useful information

  features = df.columns[:-1]
  X = df[features]
  target = df.columns[-1]
  y = df[target]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

  # Define pipeline.
  model = Pipeline([
    ('scaler', StandardScaler()), # scale to unit variance
    ('linear_regression', LinearRegression())
  ])

  # Train model.
  model.fit(X_train, y_train)

  # Save model to disk.
  dump(model, 'ml/model.skops')


if __name__ == "__main__":
  main()
