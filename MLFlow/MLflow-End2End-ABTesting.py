from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, 
    ServedEntityInput, 
    TrafficConfig, 
    Route
)

w = WorkspaceClient()

# Define the two model variants
# 'name' is a custom alias for each variant within the endpoint
served_entities = [
    ServedEntityInput(
        name="champion",
        entity_name="catalog.schema.wine_quality_prod",
        entity_version="1",
        workload_size="Small",
        scale_to_zero_enabled=True
    ),
    ServedEntityInput(
        name="challenger",
        entity_name="catalog.schema.wine_quality_test",
        entity_version="1",
        workload_size="Small",
        scale_to_zero_enabled=True
    )
]

# Define the traffic split percentage
traffic_config = TrafficConfig(
    routes=[
        Route(served_model_name="champion", traffic_percentage=90),
        Route(served_model_name="challenger", traffic_percentage=10)
    ]
)

w.serving_endpoints.update_config(
    name="wine-quality-realtime",
    served_entities=served_entities,
    traffic_config=traffic_config
)


# Note on "Sticky" A/B Testing:
# Standard traffic splitting is random per request. 
# For true A/B testing where a specific user always sees the same 
# model (stickiness), you must handle the routing logic at the 
# application level or use an external experimentation platform. 


from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import AutoCaptureConfigInput

# Update the endpoint to enable Inference Tables
w.serving_endpoints.update_config(
    name="wine-quality-realtime",
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name="main",
        schema_name="default",
        table_name_prefix="wine_model_inference"
    )
)