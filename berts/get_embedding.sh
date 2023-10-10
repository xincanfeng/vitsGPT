#!/bin/bash


# dataset="ljs"
dataset="librif"
# dataset="emovdb"

# Execute the first Python script
python3 /data/vitsGPT/berts/get_embedding_cls.py ${dataset}

# Execute the second Python script
python3 /data/vitsGPT/berts/get_embedding_text.py ${dataset}

# Execute the third Python script
python3 /data/vitsGPT/berts/get_embedding_phone.py ${dataset}
