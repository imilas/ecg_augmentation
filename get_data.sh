#!/bin/bash

# make a data folder in current directory

mkdir data
cd data


wget -O WFDB_CPSC2018.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018.tar.gz/
wget -O WFDB_CPSC2018_2.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_CPSC2018_2.tar.gz/
wget -O WFDB_StPetersburg.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining//WFDB_StPetersburg.tar.gz/
wget -O WFDB_PTB.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTB.tar.gz/
wget -O WFDB_PTBXL.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_PTBXL.tar.gz/
wget -O WFDB_Ga.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/
wget -O WFDB_ChapmanShaoxing.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ChapmanShaoxing.tar.gz/
wget -O WFDB_Ningbo.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ningbo.tar.gz/
ls *.tar.gz | xargs -i tar xf {}

# combined all 
#mkdir combined 
#find -type f -not -name "*.tar.gz" -print0 | xargs -0 cp -t combined

cd ..