from typing import NamedTuple


def import_feature_values(
    project: str,
    featurestore_id: str,
    entity_type_id: str,
    bigquery_uri: str,
    entity_id_field: str,
    bigquery_table_id: str,
    worker_count: int = 1,
    location: str = "europe-west3",
    api_endpoint: str = "europe-west3-aiplatform.googleapis.com",
    timeout: int = 500)-> NamedTuple("Outputs", [
    ("featurestore_id", str),
]): 
    import collections
    import datetime
    from google.cloud import aiplatform
    from google.protobuf.timestamp_pb2 import Timestamp
    time_now = datetime.datetime.now().timestamp()
    seconds = int(time_now)
    proto_timestamp = Timestamp(seconds=seconds)
    client_options = {"api_endpoint": api_endpoint}

    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    entity_type = f"projects/{project}/locations/{location}/featurestores/{featurestore_id}/entityTypes/{entity_type_id}"
    entity_id_field="user_id"
    
    bigquery_source = aiplatform.gapic.BigQuerySource(input_uri=bigquery_uri)

    
    feature_specs = [
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="user_id"),
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="item_id"),
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="rating"),
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="timestamp"),
    ]
    import_feature_values_request = aiplatform.gapic.ImportFeatureValuesRequest(
        entity_type=entity_type,
        bigquery_source=bigquery_source,
        feature_specs=feature_specs,
        entity_id_field=entity_id_field,
        feature_time=proto_timestamp,
        worker_count=worker_count,
        disable_online_serving=True
    )
    lro_response = client.import_feature_values(request=import_feature_values_request)
    print("Long running operation:", lro_response.operation.name)
    import_feature_values_response = lro_response.result(timeout=timeout)
    print("import_feature_values_response:", import_feature_values_response)
    
    outputs = collections.namedtuple(
      "Outputs",
      ["featurestore_id"])
    
    return outputs(featurestore_id)