import os
import json
import shutil
import argparse

def create_formated_dataset(image_dir:str, prompt:str, output_dir:str):
  dataset_dictionary = []
  dst_training_dir = output_dir + '/train/'

  # Create folder if it does not exist, remove it if it does
  if os.path.exists(dst_training_dir):
    shutil.rmtree(dst_training_dir)
  os.makedirs(dst_training_dir)

  for file_index,image in enumerate(os.listdir(image_dir)):
    # Compute image information
    image_path = os.path.join(image_dir, image)
    image_ext = image.split('.')[-1]

    # Compute new file name, dataset structure required by the training file
    index_str = str(file_index)
    while len(index_str) < 3:
      index_str = "0" + index_str
    new_file_name = index_str + '.' + image_ext

    # Copy file to training dir dst
    shutil.copyfile(image_path, dst_training_dir + new_file_name)

    # Create dataset json file, required by the training file
    dataset_dictionary.append({"file_name": "train/" + new_file_name, "text": prompt})

  # Write metadata file in dataset folder
  metadata_filepath = output_dir + "/metadata.jsonl"
  with open(metadata_filepath, 'w') as f:
    for item in dataset_dictionary:
      f.write(json.dumps(item) + "\n")

def parse_arg():
  parser = argparse.ArgumentParser(description="Script to format dataset.")
  parser.add_argument(
    "--image_dir",
    type=str,
    default=None,
    required=True,
    help="Path to images to be used for the training",
  )

  parser.add_argument(
    "--prompt",
    type=str,
    default=None,
    required=True,
    help="Prompt to be used for the training",
  )

  parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    required=True,
    help="Path to store the formated dataset",
  )

  args = parser.parse_args()
  env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
  if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

  return args

if __name__ == "__main__":
  args = parse_arg()
  create_formated_dataset(args.image_dir, args.prompt, args.output_dir)
