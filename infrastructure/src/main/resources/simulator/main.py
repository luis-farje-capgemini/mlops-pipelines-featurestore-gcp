import logging
import os
from typing import Any, Dict

import dataclasses
from google import cloud  # For patch of google.cloud.aiplatform to work.
from google.cloud import aiplatform  # For using the module.  # pylint: disable=unused-import
import tensorflow as tf  # For tf_agents to work.  # pylint: disable=unused-import
from tf_agents.bandits.environments import movielens_py_environment


@dataclasses.dataclass
class EnvVars:
  project_id: str
  region: str
  endpoint_id: str
  raw_data_path: str
  rank_k: int
  batch_size: int
  num_actions: int


def get_env_vars() -> EnvVars:
  return EnvVars(
      project_id=os.getenv("PROJECT_ID"),
      region=os.getenv("REGION"),
      endpoint_id=os.getenv("ENDPOINT_ID"),
      raw_data_path=os.getenv("RAW_DATA_PATH"),
      rank_k=int(os.getenv("RANK_K")),
      batch_size=int(os.getenv("BATCH_SIZE")),
      num_actions=int(os.getenv("NUM_ACTIONS")))


def simulate(event: Dict[str, Any], context) -> None:  # pylint: disable=unused-argument
  env_vars = get_env_vars()

  # Create MovieLens simulation environment.
  env = movielens_py_environment.MovieLensPyEnvironment(
      env_vars.raw_data_path, env_vars.rank_k, env_vars.batch_size,
      num_movies=env_vars.num_actions, csv_delimiter="\t")

  # Get environment observation.
  observation_array = env._observe()  # pylint: disable=protected-access
  # Convert to nested list to be sent to the endpoint for prediction.
  observation = [
      list(observation_batch) for observation_batch in observation_array
  ]

  cloud.aiplatform.init(
      project=env_vars.project_id, location=env_vars.region)
  endpoint = cloud.aiplatform.Endpoint(env_vars.endpoint_id)

  # Send prediction request to endpoint and get prediction result.
  predictions = endpoint.predict(
      instances=[
          {"observation": observation},
      ]
  )

  logging.info("prediction result: %s", predictions[0])
  logging.info("prediction model ID: %s", predictions[1])
