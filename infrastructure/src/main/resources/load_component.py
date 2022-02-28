from typing import NamedTuple

def load_raw_data_to_bigquery(
    project_id: str,
    raw_data_path: str,
    bigquery_dataset_id: str,
    bigquery_location: str,
    bigquery_table_id: str,
) -> NamedTuple("Outputs", [
    ("bigquery_table_id", str),
]):

  import collections
  from google.cloud import bigquery
  import logging

  def load_raw_dataset(
      project_id: str,
      bigquery_dataset_id: str,
      bigquery_location: str,
      raw_data_path: str,
      bigquery_table_id: str) -> None:
        
        client = bigquery.Client(project=project_id)
        dataset = bigquery.Dataset(bigquery_dataset_id)
        dataset.location = bigquery_location
        dataset = client.create_dataset(dataset, exists_ok=True, timeout=30)

        bigquery_table_id = bigquery_table_id
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("item_id", "STRING"),
                bigquery.SchemaField("user_id", "STRING"),
                bigquery.SchemaField("rating", "STRING"),
                bigquery.SchemaField("timestamp", "STRING"),
            ],
            source_format=bigquery.SourceFormat.CSV,
            create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            field_delimiter="\t",
        )
        uri = raw_data_path

        load_job = client.load_table_from_uri(
            uri, bigquery_table_id, job_config=job_config
        )  
        res = load_job.result()  
        logging.info(res)
        destination_table = client.get_table(bigquery_table_id) 
        logging.info("Loaded {} rows.".format(destination_table.num_rows))

  load_raw_dataset(project_id,bigquery_dataset_id, bigquery_location, raw_data_path, bigquery_table_id)

  outputs = collections.namedtuple(
      "Outputs",
      ["bigquery_table_id"])

  return outputs(bigquery_table_id)

