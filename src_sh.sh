#!/bin/bash
export  CUDA_VISIBLE_DEVICES=0
DATASET=('NWPU' 'RS' 'RSD46')
SHOT=(1)
WAY=(5)
FUSION=('att2' 'avg' 'concat' 'mfb' 'weight_avg')
WORDEMBEDDING=('bert' 'glove840B')
exp=(seq 1 10)
for i in ${exp[*]}
do
  for dataset in ${DATASET[*]}
  do
    for shot in ${SHOT[*]}
    do
      for way in ${WAY[*]}
      do
        for wordembed in ${WORDEMBEDDING[*]}
        do
          for fusion in ${FUSION[*]}
          do
  #          echo ${dataset}-${shot}-${way}-${wordembed}-${fusion} starting
            python train.py --shot ${shot} --way ${way} --dataset ${dataset}--word2vec ${wordembed} --fusionstyle ${fusion}
          done
        done
      done
    done
  done
  #
  for dataset in ${DATASET[*]}
  do
    for shot in ${SHOT[*]}
    do
      for way in ${WAY[*]}
      do
        for wordembed in ${WORDEMBEDDING[*]}
        do
          for fusion in ${[FUSION[*]]}
          do
            python train.py --shot ${shot} --way ${way} --dataset ${dataset}--word2vec ${wordembed} --fusionstyle ${fusion}
          done
        done
      done
    done
  done
  mkdir exp_${i}
  mv save/* exp_${i}
done
