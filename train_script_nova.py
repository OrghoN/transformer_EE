import json

from transformer_ee.train import MVtrainer

config_path = "transformer_ee/config/input_nova_magMomentAna_nd_fhc.json"

with open(config_path, encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

# my_trainer = MVtrainer(input_d, logger=my_logger)
my_trainer = MVtrainer(input_d)

my_trainer.train()
my_trainer.eval()
