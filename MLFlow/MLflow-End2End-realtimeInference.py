import requests
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()

endpoint_name = "wine-quality-realtime"
model_name = "wine_quality_xgboost"

# Create the endpoint
w.serving_endpoints.create(
    name=endpoint_name,
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=model_name,
                entity_version="1", # Or use 'scale_to_zero_enabled' for cost savings
                workload_size="Small",
                scale_to_zero_enabled=True
            )
        ]
    )
)



import pandas as pd

# Define a sample input row (matching your wine features)
data = {
    "dataframe_split": {
        "columns": ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
        "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
    }
}

def score_model(dataset):
    # Get the URL and Token automatically from the Databricks environment
    url = f"{dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()}/serving-endpoints/{endpoint_name}/invocations"
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    response = requests.post(url, headers=headers, data=json.dumps(dataset))
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    
    return response.json()

# Execute real-time prediction
prediction = score_model(data)
print(prediction)


import json
import requests

# 1. Prepare the payload (First 5 rows of X_test)
# 'split' orientation includes columns, index, and data separately
input_data = X_test.head(5).to_dict(orient='split')

# Wrap in the expected MLflow serving key
payload = {"dataframe_split": input_data}

# 2. Configure the Request
# Use dbutils to dynamically get the workspace URL and your personal token
endpoint_name = "wine-quality-realtime"
workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

url = f"{workspace_url}/serving-endpoints/{endpoint_name}/invocations"
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

# 3. Send the Request
response = requests.post(url, headers=headers, data=json.dumps(payload))

# 4. Check results
if response.status_code == 200:
    predictions = response.json()
    print("Real-time Predictions:", predictions)
else:
    print(f"Error {response.status_code}: {response.text}")
