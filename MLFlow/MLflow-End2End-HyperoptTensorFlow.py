import mlflow
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Broad search space: Architecture + Learning Rate
search_space = {
    'n_layers': hp.choice('n_layers', [1, 2, 3]),
    'layer_size': hp.choice('layer_size', [32, 64, 128]),
    'lr': hp.loguniform('lr', -5, -2)
}

def objective(params):
    with mlflow.start_run(nested=True):
        model = Sequential()
        model.add(Dense(params['layer_size'], activation='relu', input_shape=(X_train.shape[1],)))
        
        for _ in range(params['n_layers'] - 1):
            model.add(Dense(params['layer_size'], activation='relu'))
            
        model.add(Dense(1)) # Output layer for power prediction
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']), loss='mse')
        
        model.fit(X_train, y_train, epochs=5, validation_split=0.2, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        
        return {'loss': val_loss, 'status': STATUS_OK}

# Distributed parallel tuning across the Spark cluster
trials = SparkTrials(parallelism=4)
with mlflow.start_run(run_name="Wind_Farm_Tuning"):
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=10, trials=trials)


#add model serving end point

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput

w = WorkspaceClient()

# Create endpoint with automated monitoring (Inference Tables)
w.serving_endpoints.create(
    name="wind-turbine-monitor",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name="main.default.wind_power_forecasting",
                entity_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True
            )
        ],
        auto_capture_config=AutoCaptureConfigInput(
            catalog_name="main",
            schema_name="default",
            table_name_prefix="turbine_inference_logs"
        )
    )
)
