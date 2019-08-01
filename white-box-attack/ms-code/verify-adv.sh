#!/bin/bash

#profile_person='./profile_person'
profile_person_base='./profile_person_'

base='/home/lei/2019/AttackSpeakerVerification/data/adversarial_msasv/'
verify_file='verifymeta_adv'
subscription_key=`cat ./Verification/subscriptionkey`
persons="ph1_ ph2_ ph3_ ph4_ ph5_ ph6_ ph7_ ph8_ ph9_ ph10_"

log_file='log.verification.adv0.8.all.phrase'

cat $verify_file | while read line
do
    phrase=`echo $line | cut -d '_' -f 2`
    target_person=`echo $line | cut -d '.' -f 1 | cut -d '_' -f 4`
    phrase=$phrase"_"
    echo $target_person $phrase
    profile_id=`cat $profile_person_base$target_person | grep $phrase | cut -d ' ' -f 1`
    wav=$base$line
    echo $target_person $profile_id $wav
    cmd="python ./Verification/VerifyFile.py $subscription_key $wav $profile_id"
#    echo $cmd
    res=`$cmd`
    echo $wav $res $profile_id >> $log_file
    sleep 2s
done



