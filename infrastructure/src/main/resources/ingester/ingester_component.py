from typing import NamedTuple


def ingest_bigquery_dataset_into_tfrecord(
    project_id: str,
    bigquery_table_id: str,
    tfrecord_file: str,
    bigquery_max_rows: int = None
) -> NamedTuple("Outputs", [
    ("tfrecord_file", str),
]):

  # pylint: disable=g-import-not-at-top
  import collections
  from typing import Optional

  from google.cloud import bigquery

  import tensorflow as tf
  import logging

  def read_data_from_bigquery(
      project_id: str,
      bigquery_table_id: str,
      bigquery_max_rows: Optional[int]) -> bigquery.table.RowIterator:
   
    # Construct a BigQuery client object.
    client = bigquery.Client(project=project_id)
    

    # Get dataset.
    query_job = client.query(
        f"""
        SELECT * FROM `{bigquery_table_id}`
        """
    )
    table = query_job.result(max_results=bigquery_max_rows)
  

    return table

  def _bytes_feature(tensor: tf.Tensor) -> tf.train.Feature:
    
    value = tf.io.serialize_tensor(tensor)
    if isinstance(value, type(tf.constant(0))):
      value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def build_example(data_row: bigquery.table.Row) -> tf.train.Example:
    feature = {
        "step_type":
            _bytes_feature(data_row.get("step_type")),
        "observation":
            _bytes_feature([
                observation["observation_batch"]
                for observation in data_row.get("observation")
            ]),
        "action":
            _bytes_feature(data_row.get("action")),
        "policy_info":
            _bytes_feature(data_row.get("policy_info")),
        "next_step_type":
            _bytes_feature(data_row.get("next_step_type")),
        "reward":
            _bytes_feature(data_row.get("reward")),
        "discount":
            _bytes_feature(data_row.get("discount")),
    }
    
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto

  def write_tfrecords(
      tfrecord_file: str,
      table: bigquery.table.RowIterator) -> None:
   
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
      for data_row in table:
        example = build_example(data_row)
        writer.write(example.SerializeToString())

  table = read_data_from_bigquery(
      project_id=project_id,
      bigquery_table_id=bigquery_table_id,
      bigquery_max_rows=bigquery_max_rows)

  logging.info("writing records------------------")

  write_tfrecords(tfrecord_file, table)

  outputs = collections.namedtuple(
      "Outputs",
      ["tfrecord_file"])
  logging.info(outputs)

  return outputs(tfrecord_file)

