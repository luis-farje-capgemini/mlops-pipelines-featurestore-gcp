from google.cloud import aiplatform


def process_featurestore_list(featurestore_list):
    li = [x.name for x in featurestore_list]
    return li

def cleanup_featurestore(
    project: str,
    featurestore_id: str,
    location: str = "europe-west3",
    api_endpoint: str = "europe-west3-aiplatform.googleapis.com",
    timeout: int = 500,
):
    # The AI Platform services require regional API endpoints, which need to be
    # in the same region or multi-region overlap with the Feature Store location.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    name = client.featurestore_path(
        project=project, location=location, featurestore=featurestore_id
    )
    response = client.delete_featurestore(name=name, force=True)
    print("Long running operation:", response.operation.name)
    delete_featurestore_response = response.result()
    print("delete_featurestore_response:", delete_featurestore_response)
    

def create_featurestore(
    project: str,
    featurestore_id: str,
    location: str,
    api_endpoint: str,
    fixed_node_count: int = 1,
    timeout: int = 500,
):
    # The AI Platform services require regional API endpoints, which need to be
    # in the same region or multi-region overlap with the Feature Store location.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    parent = f"projects/{project}/locations/{location}"
    create_featurestore_request = aiplatform.gapic.CreateFeaturestoreRequest(
        parent=parent,
        featurestore_id=featurestore_id,
        featurestore=aiplatform.gapic.Featurestore(
            online_serving_config=aiplatform.gapic.Featurestore.OnlineServingConfig(
                fixed_node_count=fixed_node_count,
            ),
        ),
    )
    lro_response = client.create_featurestore(request=create_featurestore_request)
    print("Long running operation:", lro_response.operation.name)
    create_featurestore_response = lro_response.result(timeout=timeout)
    print("create_featurestore_response:", create_featurestore_response)


    
def list_featurestore(
    project: str,
    location: str,
    api_endpoint: str,
    timeout: int = 500,
):
    # The AI Platform services require regional API endpoints, which need to be
    # in the same region or multi-region overlap with the Feature Store location.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    parent = f"projects/{project}/locations/{location}"
    # create_featurestore_request = aiplatform.gapic.ListFeaturestoresRequest(parent=parent)
    featurestore_list = client.list_featurestores(parent=parent)
    featurestore_list = process_featurestore_list(featurestore_list)
    print(f"Featurestores found: {len(featurestore_list)}")
    return featurestore_list




def create_entity_type(
    project: str,
    featurestore_id: str,
    entity_type_id: str,
    location: str,
    api_endpoint: str,
    description: str = "entity",
    timeout: int = 300,
):
    # The AI Platform services require regional API endpoints, which need to be
    # in the same region or multi-region overlap with the Feature Store location.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    parent = f"projects/{project}/locations/{location}/featurestores/{featurestore_id}"
    create_entity_type_request = aiplatform.gapic.CreateEntityTypeRequest(
        parent=parent,
        entity_type_id=entity_type_id,
        entity_type=aiplatform.gapic.EntityType(description=description),
    )
    lro_response = client.create_entity_type(request=create_entity_type_request)
    print("Long running operation:", lro_response.operation.name)
    create_entity_type_response = lro_response.result(timeout=timeout)
    print("create_entity_type_response:", create_entity_type_response)

def create_feature(
    project: str,
    featurestore_id: str,
    entity_type_id: str,
    feature_id: str,
    value_type: aiplatform.gapic.Feature.ValueType,
    location: str,
    api_endpoint: str,
    timeout: int = 300,
    description: str = "feature",
):
    # The AI Platform services require regional API endpoints, which need to be
    # in the same region or multi-region overlap with the Feature Store location.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    parent = f"projects/{project}/locations/{location}/featurestores/{featurestore_id}/entityTypes/{entity_type_id}"
    create_feature_request = aiplatform.gapic.CreateFeatureRequest(
        parent=parent,
        feature=aiplatform.gapic.Feature(
            value_type=value_type, description=description
        ),
        feature_id=feature_id,
    )
    lro_response = client.create_feature(request=create_feature_request)
    print("Long running operation:", lro_response.operation.name)
    create_feature_response = lro_response.result(timeout=timeout)
    print("create_feature_response:", create_feature_response)
    
from google.cloud import aiplatform


def import_feature_values(
    project: str,
    featurestore_id: str,
    entity_type_id: str,
    bigquery_uri: str,
    entity_id_field: str,
    location: str,
    api_endpoint: str,
    worker_count: int = 1,
    timeout: int = 500,
):
    
    import datetime
    from google.protobuf.timestamp_pb2 import Timestamp
    time_now = datetime.datetime.now().timestamp()
    seconds = int(time_now)
    proto_timestamp = Timestamp(seconds=seconds)

    client_options = {"api_endpoint": api_endpoint}

    client = aiplatform.gapic.FeaturestoreServiceClient(client_options=client_options)
    entity_type = f"projects/{project}/locations/{location}/featurestores/{featurestore_id}/entityTypes/{entity_type_id}"
    entity_id_field="user_id"
    
    feature_specs = [
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="user_id"),
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="item_id"),
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="rating"),
        aiplatform.gapic.ImportFeatureValuesRequest.FeatureSpec(id="timestamp"),
    ]
    bigquery_src = aiplatform.gapic.BigQuerySource("bq://mlops-insights-data-sweden.movielens_dataset.training_dataset")
    import_feature_values_request = aiplatform.gapic.ImportFeatureValuesRequest(
        entity_type=entity_type,
        bigquery_source=bigquery_src,
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

