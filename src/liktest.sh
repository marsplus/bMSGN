#!/bin/bash

nAgent=100


if [[ $1 == "simulation" ]]
then
    ### generate synthetic data
    for gType in BA SW BTER
    do
       for i in {1..30}
       do
           for tStep in 250 500 750 1000 1250 1500 
           #for tStep in 2000
               do
                   python simulate.py --timeStep=$tStep --numAgent=$nAgent --graph_type=$gType --seed=$i --liktest=1 --noGroup=1
                   python simulate.py --timeStep=$tStep --numAgent=$nAgent --graph_type=$gType --seed=$i --liktest=1 --noGroup=0
               done
               BACK_PID=$!
               wait $BACK_PID
       done
    done

elif [[ $1 == "liktest"  ]]
    ### the actual likelihood test
    for i in {1..30}
    do
        for tStep in 250 500 750 1000 1250 1500
        do
            for gType in BA SW BTER
            do
            	for mode in noGroup withGroup
            	do
            		fPath="../result/LikRatio/Data_timeStep_${tStep}_gamma_5.00_numAgent_${nAgent}_graphType_${gType}_seed_${i}_${mode}.p"
                	python liktest.py --timeStep=$tStep --numTrain=$tStep --numTest=0 --seed=$i --graph_type=$gType --fPath=$fPath >> ../result/liktest_${mode}.txt &     
                done
                BACK_PID=$!
            	wait $BACK_PID 
            done
        done
    done
else
    echo "unknown mode."
fi