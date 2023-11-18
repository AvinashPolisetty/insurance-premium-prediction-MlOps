# insurance-premium-prediction-MlOps

MLFLOW_TRACKING_URI=https://dagshub.com/avinash8/insurance-premium-prediction-MlOps.mlflow \
MLFLOW_TRACKING_USERNAME=avinash8 \
MLFLOW_TRACKING_PASSWORD=d7441ad4dd1b8e54aec5de0dc81c53c783cb7d63 \
python script.py



def log_into_mlflow(self):
        
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            pred=model.predict(test_x)

            (rmse,mae,r2)=self.eval_metrics(test_y,pred)
             # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("r2",r2)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
