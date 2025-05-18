from skops.io import load, get_untrusted_types
from sklearn.utils.validation import check_is_fitted


def test_model_loads_correctly():
  model_path = 'ml/model.skops'
  unknown_types = get_untrusted_types(file=model_path)
  model = load(model_path, trusted=unknown_types)

  assert model is not None
  assert hasattr(model, 'predict')
  check_is_fitted(model)
  