#!/bin/bash

#dataset=Electronics
#dataset=Clothing_Shoes_and_Jewelry
#dataset=Books
#dataset=Sports_and_Outdoors
dataset=Toys_and_Games

mkdir stats 
mkdir data 
mkdir tmp
mkdir dgl_graph

echo "---------------- step 1: edge extraction ---------------"
python edge_extractor.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 2: data formulation --------------"
python data_formulator.py $dataset
echo "--------------------------------------------------------"

