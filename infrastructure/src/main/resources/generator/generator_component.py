from typing import NamedTuple


def generate_movielens_dataset_for_bigquery(
    project_id: str,
    raw_data_path: str,
    batch_size: int,
    rank_k: int,
    num_actions: int,
    driver_steps: int,
    bigquery_tmp_file: str,
    bigquery_dataset_id: str,
    bigquery_location: str,
    bigquery_table_id: str
) -> NamedTuple("Outputs", [
    ("bigquery_dataset_id", str),
    ("bigquery_location", str),
    ("bigquery_table_id", str),
]):
 
  # pylint: disable=g-import-not-at-top
  import collections
  import json
  from typing import Any, Dict

  from google.cloud import bigquery

  from tf_agents import replay_buffers
  from tf_agents import trajectories
  from tf_agents.bandits.agents.examples.v2 import trainer
  from tf_agents.bandits.environments import movielens_py_environment
  from tf_agents.drivers import dynamic_step_driver
  from tf_agents.environments import tf_py_environment
  from tf_agents.policies import random_tf_policy

  def generate_simulation_data(
      raw_data_path: str,
      batch_size: int,
      rank_k: int,
      num_actions: int,
      driver_steps: int) -> replay_buffers.TFUniformReplayBuffer:

    # Create movielens simulation environment.
    env = movielens_py_environment.MovieLensPyEnvironment(
        raw_data_path,
        rank_k,
        batch_size,
        num_movies=num_actions,
        csv_delimiter="\t")
    environment = tf_py_environment.TFPyEnvironment(env)

    # Define random policy for collecting data.
    random_policy = random_tf_policy.RandomTFPolicy(
        action_spec=environment.action_spec(),
        time_step_spec=environment.time_step_spec())

    # Use replay buffer and observers to keep track of Trajectory data.
    data_spec = random_policy.trajectory_spec
    replay_buffer = trainer.get_replay_buffer(data_spec, environment.batch_size,
                                              driver_steps)
    observers = [replay_buffer.add_batch]

    # Run driver to apply the random policy in the simulation environment.
    driver = dynamic_step_driver.DynamicStepDriver(
        env=environment,
        policy=random_policy,
        num_steps=driver_steps * environment.batch_size,
        observers=observers)
    driver.run()

    return replay_buffer

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

  def write_replay_buffer_to_file(
      replay_buffer: replay_buffers.TFUniformReplayBuffer,
      batch_size: int,
      dataset_file: str) -> None:
    
    dataset = replay_buffer.as_dataset(sample_batch_size=batch_size)
    dataset_size = replay_buffer.num_frames().numpy()

    with open(dataset_file, "w") as f:
      for example in dataset.take(count=dataset_size):
        traj_dict = build_dict_from_trajectory(example[0])
        f.write(json.dumps(traj_dict) + "\n")

  def load_dataset_into_bigquery(
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
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    with open(dataset_file, "rb") as source_file:
      load_job = client.load_table_from_file(
          source_file, bigquery_table_id, job_config=job_config)

    load_job.result()  # Wait for the job to complete.

  replay_buffer = generate_simulation_data(
      raw_data_path=raw_data_path,
      batch_size=batch_size,
      rank_k=rank_k,
      num_actions=num_actions,
      driver_steps=driver_steps)

  write_replay_buffer_to_file(
      replay_buffer=replay_buffer,
      batch_size=batch_size,
      dataset_file=bigquery_tmp_file)

  load_dataset_into_bigquery(project_id, bigquery_tmp_file, bigquery_dataset_id,
                             bigquery_location, bigquery_table_id)

  outputs = collections.namedtuple(
      "Outputs",
      ["bigquery_dataset_id", "bigquery_location", "bigquery_table_id"])

  return outputs(bigquery_dataset_id, bigquery_location, bigquery_table_id)
