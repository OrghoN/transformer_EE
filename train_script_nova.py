import json

from transformer_ee.train import MVtrainer

with open("transformer_ee/config/input_nova_mprod6_1_OPAL_nd_fhc.json", encoding="UTF-8", mode="r") as f:
    input_d = json.load(f)

# my_trainer = MVtrainer(input_d, logger=my_logger)
my_trainer = MVtrainer(input_d)

my_trainer.train()
my_trainer.eval()
