from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('app_request_count', 'Total requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency', ['endpoint'])

PREPROCESSING_TIME = Histogram('data_preprocessing_seconds', 'Time spent in to_dataframe')
MODEL_INFERENCE_TIME = Histogram('model_inference_time_seconds', 'Time spent predicting')
MODEL_PROBABILITY = Histogram('model_prediction_probability', 'Distribution of probabilities')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total predictions by class', ['prediction_class'])

MODEL_INFO = Gauge('model_info', 'Current model info', ['run_id', 'model_type', 'features_count'])