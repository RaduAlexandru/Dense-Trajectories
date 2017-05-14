#!/bin/bash


#block(name=extract-features, threads=1, memory=10000, gpus=0, hours=4)

    mkdir -p log results

    for PART in test train; do

        OPTIONS="--log-file=log/$PART.stip.log \
                 --features.feature-writer.feature-cache=results/$PART.stip.gz \
                 --stip.video-list=$PART.videos"

        ../../src/ActionRecognition/action-recognizer --config=config/stip.config $OPTIONS

    done


#block(name=train, threads=4, memory=10000, gpus=0, hours=24)

    ../../src/ActionRecognition/action-recognizer --config=config/svm-train.config


#block(name=classify, threads=1, memory=10000, gpus=0, hours=4)

    ../../src/ActionRecognition/action-recognizer --config=config/svm-classify.config

