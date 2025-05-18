### What approach would you use to version and track different models in production?

I would use a model registry to store each model along with its metadata, including the version of the data it was trained on, hyperparameters used in training, learned parameters, performance metrics, a description of code changes, etc.
Each model would naturally be assigned a unique version.
Besides traceability and reproducibility, this also enables structured comparisons between models and optimization experiments, for example.
I would track code, dependencies, and environment configurations via source control (Git) and integrate it into a CI/CD pipeline to automatically validate models before deployment.
Using containerization tools would help ensure consistency across environments.

After deployment, I would use monitoring to track model performance over time to detect degradation caused, for example, by data or concept drift, to ensure models stay accurate and reliable.
The versioning would allow performance issues to be traced back to specific models or datasets and, for example, trigger retraining.
A platform such as MLflow would enable all of this, since it supports management of the full model lifecycle.


### What key metrics would you monitor for this API service and the prediction model?

Since I've deployed a linear regression model, I would monitor key regression performance metrics such as mean absolute error (MAE), (root) mean squared error (MSE), and R-squared score. 
Since true labels are not always immediately available (not until the house is sold, in this particular case), representative reference data might have to be used to compute these metrics.
I would also track indicators of data drift (does the distribution of the input features change over time?) and output drift (does the distribution of predicted prices suddenly change?). 
Such drift can cause performance degradation and should potentially trigger retraining. 
Finer-grained performance is important to monitor as well, since aggregate metrics can hide underperformance in specific regions of the data.
I would also monitor input data quality (missing values, type mismatches, etc.) and log predictions along with their inputs.
Tools like SHAP can support explainability if model transparancy is required, although a linear regression model is inherently interpretable, since it just computes the inner product between the learned weights and the input features.

At the API service level, I would track operational metrics such as prediction latency, memory usage, error rates, etc., to ensure that the model is reliable. 
This would include logging incoming requests and prediction responses.
