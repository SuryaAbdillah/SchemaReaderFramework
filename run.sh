#!/bin/bash

#--input-data /home/alim/Advanced-DB/github-event/${f}/github_events_v2_${f}.jsonl \
#--input-data /home/alim/Advanced-DB/sensor_data_more_features/sensor_data_more_features_${f}_valid.jsonl \
for f in 10k; do
  python main.py --mode benchmark \
    --chunk-size 1000 \
    --input-data /home/alim/Advanced-DB/sensor_data_more_features/sensor_data_more_features_${f}_valid.jsonl \
    --num-workers 10 \
    --repeat 10 \
    --formats jsonl csv parquet avro feather \
    --output-dir output/
done
