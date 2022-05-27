#!/bin/bash

nAgent=100


if [[ $1 == "simulation" ]]
then
    ## generate synthetic data
    for gType in BA SW BTER
    do
        for i in {1..30}
        do
            for tStep in 250 500 750 1000 1250 1500 
            #for tStep in 2000
                do
                    python simulate.py --timeStep=$tStep --numAgent=$nAgent --graph_type=$gType --seed=$i 
                done
        done
    done

elif [[ $1 == "estimation" ]]
    ## run the estimation
    for i in {1..30}
    do
        for tStep in 250 500 750 1000 1250 1500
        do
            for gType in BA SW BTER
            do
                python main.py --timeStep=$tStep --numTrain=$tStep --numTest=0 --seed=$i --graph_type=$gType >> ../result/trend_${gType}.txt     
            done
            #BACK_PID=$!
            #wait $BACK_PID 
        done
    done
else
    echo "unknown mode."
fi
