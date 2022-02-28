from typing import NamedTuple  
from kfp import components


def training_op(
    training_artifacts_dir: str,
    tfrecord_file: str,
    num_epochs: int,
    rank_k: int,
    num_actions: int,
    tikhonov_weight: float,
    agent_alpha: float
) -> NamedTuple("Outputs", [
    ("training_artifacts_dir", components.OutputPath),
]):
  import collections
  from typing import Dict, List, NamedTuple 

  import tensorflow as tf

  from tf_agents import agents
  from tf_agents import policies
  from tf_agents import trajectories
  from tf_agents.bandits.agents import lin_ucb_agent
  from tf_agents.policies import policy_saver
  from tf_agents.specs import tensor_spec

  import logging

  per_arm = False  

  feature_description = {
      "step_type": tf.io.FixedLenFeature((), tf.string),
      "observation": tf.io.FixedLenFeature((), tf.string),
      "action": tf.io.FixedLenFeature((), tf.string),
      "policy_info": tf.io.FixedLenFeature((), tf.string),
      "next_step_type": tf.io.FixedLenFeature((), tf.string),
      "reward": tf.io.FixedLenFeature((), tf.string),
      "discount": tf.io.FixedLenFeature((), tf.string),
  }

  def _parse_record(raw_record: tf.Tensor) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(raw_record, feature_description)

  def build_trajectory(
      parsed_record: Dict[str, tf.Tensor],
      policy_info: policies.utils.PolicyInfo) -> trajectories.Trajectory:
    return trajectories.Trajectory(
        step_type=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["step_type"], out_type=tf.int32),
            axis=1),
        observation=tf.expand_dims(
            tf.io.parse_tensor(
                parsed_record["observation"], out_type=tf.float32),
            axis=1),
        action=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["action"], out_type=tf.int32),
            axis=1),
        policy_info=policy_info,
        next_step_type=tf.expand_dims(
            tf.io.parse_tensor(
                parsed_record["next_step_type"], out_type=tf.int32),
            axis=1),
        reward=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["reward"], out_type=tf.float32),
            axis=1),
        discount=tf.expand_dims(
            tf.io.parse_tensor(parsed_record["discount"], out_type=tf.float32),
            axis=1))

  def train_policy_on_trajectory(
      agent: agents.TFAgent,
      tfrecord_file: str,
      num_epochs: int
  ) -> NamedTuple("TrainOutputs", [
      ("policy", policies.TFPolicy),
      ("train_loss", Dict[str, List[float]]),
  ]):
    raw_dataset = tf.data.TFRecordDataset([tfrecord_file])
    parsed_dataset = raw_dataset.map(_parse_record)

    train_loss = collections.defaultdict(list)
    for epoch in range(num_epochs):
      for parsed_record in parsed_dataset:
        trajectory = build_trajectory(parsed_record, agent.policy.info_spec)
        loss, _ = agent.train(trajectory)
        train_loss[f"epoch{epoch + 1}"].append(loss.numpy())

    train_outputs = collections.namedtuple(
        "TrainOutputs",
        ["policy", "train_loss"])
    return train_outputs(agent.policy, train_loss)

  def execute_training_and_save_policy(
      training_artifacts_dir: str,
      tfrecord_file: str,
      num_epochs: int,
      rank_k: int,
      num_actions: int,
      tikhonov_weight: float,
      agent_alpha: float) -> None:
    # Define time step and action specs for one batch.
    time_step_spec = trajectories.TimeStep(
        step_type=tensor_spec.TensorSpec(
            shape=(), dtype=tf.int32, name="step_type"),
        reward=tensor_spec.TensorSpec(
            shape=(), dtype=tf.float32, name="reward"),
        discount=tensor_spec.BoundedTensorSpec(
            shape=(), dtype=tf.float32, name="discount", minimum=0.,
            maximum=1.),
        observation=tensor_spec.TensorSpec(
            shape=(rank_k,), dtype=tf.float32,
            name="observation"))

    action_spec = tensor_spec.BoundedTensorSpec(
        shape=(),
        dtype=tf.int32,
        name="action",
        minimum=0,
        maximum=num_actions - 1)

    # Define RL agent/algorithm.
    agent = lin_ucb_agent.LinearUCBAgent(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        tikhonov_weight=tikhonov_weight,
        alpha=agent_alpha,
        dtype=tf.float32,
        accepts_per_arm_features=per_arm)
    agent.initialize()
    logging.info("TimeStep Spec (for each batch):\n%s\n", agent.time_step_spec)
    logging.info("Action Spec (for each batch):\n%s\n", agent.action_spec)

    # Perform off-policy training.
    policy, _ = train_policy_on_trajectory(
        agent=agent,
        tfrecord_file=tfrecord_file,
        num_epochs=num_epochs)

    # Save trained policy.
    saver = policy_saver.PolicySaver(policy)
    saver.save("gs://mlops-vertex-capgemini/artifacts")

  execute_training_and_save_policy(
      training_artifacts_dir=training_artifacts_dir,
      tfrecord_file=tfrecord_file,
      num_epochs=num_epochs,
      rank_k=rank_k,
      num_actions=num_actions,
      tikhonov_weight=tikhonov_weight,
      agent_alpha=agent_alpha)

  outputs = collections.namedtuple(
      "Outputs",
      ["training_artifacts_dir"])

  return outputs(training_artifacts_dir)