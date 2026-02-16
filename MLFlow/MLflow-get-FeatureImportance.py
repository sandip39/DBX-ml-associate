import mlflow
import pandas as pd

# Define the model path or name (e.g., "models:/<model_name>/<stage>" or a run-relative path)
model_path = "models:/my_registered_model/staging" 

# Load the model as a generic Python function
# This loads the 'pyfunc' flavor of the model
loaded_model = mlflow.pyfunc.load_model(model_path)

# Access the underlying native model (assuming it's a model with feature_importances_)
# The exact attribute might vary based on how the model was logged.
try:
    native_model = loaded_model._model_impl.python_model.model
    
    # Get feature names (you might need to log these separately if not available in the model)
    # This example assumes feature_name_ is an attribute of the model
    feature_names = getattr(native_model, 'feature_name_', None) 
    
    if feature_names is None:
        # Fallback or manual list of feature names if not logged with the model
        # Replace with your actual feature names
        feature_names = ["feature_1", "feature_2", "..."] 

    # Get the feature importances
    importances = native_model.feature_importances_

    # Create a pandas DataFrame for better viewing and sorting
    feature_importance_df = pd.DataFrame({'feature_name': feature_names, 'importance': importances})
    
    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
    
    # Display the top 10 features
    print("Top 10 features:")
    print(feature_importance_df.head(10))

except AttributeError as e:
    print(f"Could not access feature importances directly from the model: {e}")
    print("Ensure your model type supports '.feature_importances_' or log them as artifacts.")

