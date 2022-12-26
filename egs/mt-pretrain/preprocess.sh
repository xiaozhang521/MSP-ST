DICT= # path to mbart dict
prefix=
SRC=en_XX
#TGT=de_DE
TGT=fr_XX
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref $prefix/  \ #traing file
  --testpref  $prefix/ \ #valid file 
  --validpref $prefix/ \  #test file
  --destdir #path to save data \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 24

