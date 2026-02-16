import os
from databricks.sdk import WorkspaceClient

# Set these in your environment for security
w = WorkspaceClient(
    host="https://<workspace-url>",
    client_id=os.environ.get("DATABRICKS_CLIENT_ID"),
    client_secret=os.environ.get("DATABRICKS_CLIENT_SECRET")
)

# Send an inference request to the secured endpoint
response = w.serving_endpoints.query(
    name="wine-quality-realtime",
    dataframe_split={
        "columns": ["fixed acidity", "volatile acidity"],
        "data": [[7.4, 0.7]]
    }
)
