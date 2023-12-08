# TODO: these give me the tuned hypers for SET single subject, let's plot them
# now!
#
# RIGHT NOW WORKING ON combine_by_indices.py
import sparseeg.util.hyper as hyper
import yaml
import json
from pprint import pprint

# hyper_file = "./plot/set_single_subject_hypers.json"
hyper_file = "./plot/wp_hypers_combined_5_seeds.json"
with open(hyper_file, "r") as i:
    hypers = json.load(i)

config_file = (
    "./src/sparseeg/config/eeg_low_final/weighted_loss/" +
    "single_subject/wp_500epochs.yaml"
)
# config_file = (
#     "./src/sparseeg/config/eeg_low_final/weighted_loss/" +
#     "single_subject/wp_500epochs_next_5_seeds.yaml"
# )

with open(config_file, "r") as i:
    config = yaml.safe_load(i)

print("Total:", hyper.total(config))
print("Best:")
for h in hypers:
    print("\t", hyper.index_of(config, h, ignore=["seed", "save_dir"]))
