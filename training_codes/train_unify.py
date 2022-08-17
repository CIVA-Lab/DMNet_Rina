import sys
import os
sys.path.insert(0, 'solaris_rina')
print (sys.path)
sol = __import__('solaris_rina')


# Dataset location (edit as needed)
dataset_namein = sys.argv[1]
gttype=sys.argv[2]
branch=sys.argv[3]
config_path=sys.argv[4]
# Load config
config = sol.utils.config.parse(config_path)


# %% ============================
# Training
# ===============================

# make model output dir



os.makedirs(os.path.dirname(config['training']['model_dest_path']), exist_ok=True)
config['pretrained_rina']=True
#if gttype in ["GT"]:
#    config['training']['epochs']=100
#from sol.nets import datagen_cell
trainer = sol.nets.train_cell_GTST.Trainer(config=config,dataset_name=dataset_namein,branch=branch,GT=gttype)

trainer.train()