import json
import os
from typing import Dict, List

import fastapi

from google.cloud import pubsub_v1

import tensorflow as tf
import tf_agents
from tf_agents import policies
import tensorflow_probability as tfp
tfd = tfp.distributions


app = fastapi.FastAPI()
app_vars = {"trained_policy": None}


def _startup_event() -> None:
  app_vars["trained_policy"] = tf.saved_model.load(
      os.environ["AIP_STORAGE_URI"])

@app.on_event("startup")
async def startup_event() -> None:
  _startup_event()


def _health() -> Dict[str, str]:
  return {}


@app.get(os.environ["AIP_HEALTH_ROUTE"], status_code=200)
def health() -> Dict[str, str]:
  return _health()


def _message_logger_via_pubsub(
    project_id: str,
    logger_pubsub_topic: str,
    observations: List[Dict[str, List[List[float]]]],
    predicted_actions: List[Dict[str, List[float]]]) -> None:
  # Create message with observations and predicted actions.
  message_json = json.dumps({
      "observations": observations,
      "predicted_actions": predicted_actions,
  })
  message_bytes = message_json.encode("utf-8")

  # Instantiate a Pub/Sub client.
  publisher = pubsub_v1.PublisherClient()

  # Get the Logger's Pub/Sub topic.
  topic_path = publisher.topic_path(project_id, logger_pubsub_topic)

  # Send message.
  publish_future = publisher.publish(topic_path, data=message_bytes)
  publish_future.result()


def _predict(
    instances: List[Dict[str, List[List[float]]]],
    trained_policy: policies.TFPolicy) -> Dict[str, List[Dict[str, List[int]]]]:
  predictions = []
  predicted_actions = []
  for index, instance in enumerate(instances):
    # Unpack observation and reconstruct TimeStep. Rewards default to 0.
    batch_size = len(instance["observation"])
    time_step = tf_agents.trajectories.restart(
        observation=instance["observation"],
        batch_size=tf.convert_to_tensor([batch_size]))
    policy_step = trained_policy.action(time_step)

    predicted_action = policy_step.action.numpy().tolist()
    predictions.append(
        {f"PolicyStep {index}": predicted_action})
    predicted_actions.append({"predicted_action": predicted_action})

  # Trigger the Logger to log prediction inputs and results.
  _message_logger_via_pubsub(
      project_id=os.environ["PROJECT_ID"],
      logger_pubsub_topic=os.environ["LOGGER_PUBSUB_TOPIC"],
      observations=instances,
      predicted_actions=predicted_actions)
  return {"predictions": predictions}


@app.post(os.environ["AIP_PREDICT_ROUTE"])
async def predict(
    request: fastapi.Request) -> Dict[str, List[Dict[str, List[int]]]]:
  body = await request.json()
  instances = body["instances"]
  return _predict(instances, app_vars["trained_policy"])
