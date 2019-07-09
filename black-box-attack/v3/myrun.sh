#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_trials=data/voxceleb1_test/trials
#voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb1_root=/home/lei/dataset/voxceleb2/vox/
#voxceleb2_root=/export/corpora/VoxCeleb2
nnet_dir=exp/xvector_nnet_1a
#musan_root=/export/corpora/JHU/musan

stage=15

if [ $stage -le 0 ]; then
  #local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  #local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test
  
  # This script reates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1.pl $voxceleb1_root data
  # We'll train on all of VoxCeleb2, plus the training portion of VoxCeleb1.
  # This should give 7,351 speakers and 1,277,503 utterances.
  
  #utils/combine_data.sh data/train data/voxceleb2_train data/voxceleb2_test data/voxceleb1_train
  utils/combine_data.sh data/train  data/voxceleb1_train
fi


if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  #for name in train voxceleb1_test; do
  for name in voxceleb1_test; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi


if [ $stage -le 9 ]; then
  # Extract x-vectors for centering, LDA, and PLDA training.
  #sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 80 \
  #  $nnet_dir data/train \
  #  $nnet_dir/xvectors_train

  # Extract x-vectors used in the evaluation.
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
    $nnet_dir data/voxceleb1_test \
    $nnet_dir/xvectors_voxceleb1_test
fi


if [ $stage -le 11 ]; then
    # computing the score for the trials
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$nnet_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
fi

if [ $stage -le 15 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.224%
  # minDCF(p-target=0.01): 0.3492
  # minDCF(p-target=0.001): 0.5452
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.419%
  # minDCF(p-target=0.01): 0.4701
  # minDCF(p-target=0.001): 0.5981
fi
