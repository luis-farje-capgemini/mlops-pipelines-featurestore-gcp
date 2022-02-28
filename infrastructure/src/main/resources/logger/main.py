import base64
import dataclasses
import json
import os
import tempfile
from typing import Any, Dict, List

from google.cloud import bigquery
import tensorflow as tf
from tf_agents import trajectories
from tf_agents.bandits.environments import movielens_py_environment
from tf_agents.environments import tf_py_environment


@dataclasses.dataclass
class EnvVars:
  project_id: str
  raw_data_path: str
  batch_size: int
  rank_k: int
  num_actions: int
  bigquery_tmp_file: str
  bigquery_dataset_id: str
  bigquery_location: str
  bigquery_table_id: str


def get_env_vars() -> EnvVars:
  return EnvVars(
      project_id=os.getenv("PROJECT_ID"),
      raw_data_path=os.getenv("RAW_DATA_PATH"),
      batch_size=int(os.getenv("BATCH_SIZE")),
      rank_k=int(os.getenv("RANK_K")),
      num_actions=int(os.getenv("NUM_ACTIONS")),
      bigquery_tmp_file=os.getenv("BIGQUERY_TMP_FILE"),
      bigquery_dataset_id=os.getenv("BIGQUERY_DATASET_ID"),
      bigquery_location=os.getenv("BIGQUERY_LOCATION"),
      bigquery_table_id=os.getenv("BIGQUERY_TABLE_ID"))


def replace_observation_in_time_step(
    original_time_step: trajectories.TimeStep,
    observation: tf.Tensor) -> trajectories.TimeStep:
  return trajectories.TimeStep(
      step_type=original_time_step[0],
      reward=original_time_step[1],
      discount=original_time_step[2],
      observation=observation)


def get_trajectory_from_environment(
    environment: tf_py_environment.TFPyEnvironment,
    observation: List[List[float]],
    predicted_action: int) -> trajectories.Trajectory:
  environment.reset()

  # Align environment to observation.
  original_time_step = environment.current_time_step()
  time_step = replace_observation_in_time_step(original_time_step, observation)
  environment._time_step = time_step  # pylint: disable=protected-access

  # Apply predicted action to environment.
  environment.step(action=predicted_action)

  # Get next time step.
  next_time_step = environment.current_time_step()

  # Get trajectory as an encapsulation of all feedback from the environment.
  trajectory = trajectories.from_transition(
      time_step=time_step,
      action_step=trajectories.PolicyStep(action=predicted_action),
      next_time_step=next_time_step)
  return trajectory


def build_dict_from_trajectory(
    trajectory: trajectories.Trajectory) -> Dict[str, Any]:
  trajectory_dict = {
      "step_type": trajectory.step_type.numpy().tolist(),
      "observation": [{
          "observation_batch": batch
      } for batch in trajectory.observation.numpy().tolist()],
      "action": trajectory.action.numpy().tolist(),
      "policy_info": trajectory.policy_info,
      "next_step_type": trajectory.next_step_type.numpy().tolist(),
      "reward": trajectory.reward.numpy().tolist(),
      "discount": trajectory.discount.numpy().tolist(),
  }
  return trajectory_dict


def write_trajectories_to_file(
    dataset_file: str,
    environment: tf_py_environment.TFPyEnvironment,
    observations: List[Dict[str, List[List[float]]]],
    predicted_actions: List[Dict[str, List[float]]]) -> None:
  with open(dataset_file, "w") as f:
    for observation, predicted_action in zip(observations, predicted_actions):
      trajectory = get_trajectory_from_environment(
          environment=environment,
          observation=tf.constant(observation["observation"]),
          predicted_action=tf.constant(predicted_action["predicted_action"]))
      trajectory_dict = build_dict_from_trajectory(trajectory)
      f.write(json.dumps(trajectory_dict) + "\n")


def append_dataset_to_bigquery(
    project_id: str,
    dataset_file: str,
    bigquery_dataset_id: str,
    bigquery_location: str,
    bigquery_table_id: str) -> None:
  # Construct a BigQuery client object.
  client = bigquery.Client(project=project_id)

  # Construct a full Dataset object to send to the API.
  dataset = bigquery.Dataset(bigquery_dataset_id)

  # Specify the geographic location where the dataset should reside.
  dataset.location = bigquery_location

  # Create the dataset, or get the dataset if it exists.
  dataset = client.create_dataset(dataset, exists_ok=True, timeout=30)

  job_config = bigquery.LoadJobConfig(
      write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
      schema=[
          bigquery.SchemaField("step_type", "INT64", mode="REPEATED"),
          bigquery.SchemaField(
              "observation",
              "RECORD",
              mode="REPEATED",
              fields=[
                  bigquery.SchemaField("observation_batch", "FLOAT64",
                                       "REPEATED")
              ]),
          bigquery.SchemaField("action", "INT64", mode="REPEATED"),
          bigquery.SchemaField("policy_info", "FLOAT64", mode="REPEATED"),
          bigquery.SchemaField("next_step_type", "INT64", mode="REPEATED"),
          bigquery.SchemaField("reward", "FLOAT64", mode="REPEATED"),
          bigquery.SchemaField("discount", "FLOAT64", mode="REPEATED"),
      ],
      source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
  )

  with open(dataset_file, "rb") as source_file:
    load_job = client.load_table_from_file(
        source_file, bigquery_table_id, job_config=job_config)

  load_job.result()  # Wait for the job to complete.


def log_prediction_to_bigquery(event: Dict[str, Any], context) -> None:  # pylint: disable=unused-argument
  env_vars = get_env_vars()
  # Get a file path with permission for writing.
  dataset_file = os.path.join(tempfile.gettempdir(), env_vars.bigquery_tmp_file)

  data_bytes = base64.b64decode(event["data"])
  data_json = data_bytes.decode("utf-8")
  data = json.loads(data_json)
  observations = data["observations"]
  predicted_actions = data["predicted_actions"]

  # Create MovieLens simulation environment.
  env = movielens_py_environment.MovieLensPyEnvironment(
      env_vars.raw_data_path,
      env_vars.rank_k,
      env_vars.batch_size,
      num_movies=env_vars.num_actions,
      csv_delimiter="\t")
  environment = tf_py_environment.TFPyEnvironment(env)

  # Get environment feedback and write trajectory data.
  write_trajectories_to_file(
      dataset_file=dataset_file,
      environment=environment,
      observations=observations,
      predicted_actions=predicted_actions)

  # Add trajectory data as new training data to BigQuery.
  append_dataset_to_bigquery(
      project_id=env_vars.project_id,
      dataset_file=dataset_file,
      bigquery_dataset_id=env_vars.bigquery_dataset_id,
      bigquery_location=env_vars.bigquery_location,
      bigquery_table_id=env_vars.bigquery_table_id)
