#!/bin/bash
# for each profile id in profile.txt, select a corresponding person in ../data/vad_train/, enroll for max time, ( need to be checked) , write the corresponding information in a file
#list=`cat ./profile_person_test`

#profile_person='./profile_person'
profile_person='./profile_person_lei'

base='/home/lei/2019/AttackSpeakerVerification/data/adversarial_msasv/'
#base='/home/lei/2019/AttackSpeakerVerification/data/suibin/pcm16/'
#verify_file='verifymeta_adv'verifymeta_adv_suibin_lei'
verify_file='verifymeta_adv_suibin_lei'
#verify_file='verifymeta_ori_suibin_lei'
subscription_key=`cat ./Verification/subscriptionkey`
persons="ph1_ ph2_ ph3_ ph4_ ph5_ ph6_ ph7_ ph8_ ph9_ ph10_"

#log_file='verification_log_adv_with_single_embedding_06'
log_file='verification_log_adv_with_single_embedding_06_suibin_lei'

cat $verify_file | while read line
do
    person=`echo $line | cut -d '_' -f 2`
    person=$person"_"
    profile_id=`cat $profile_person | grep $person | cut -d ' ' -f 1`
    #wav=`echo $line | cut -d ' ' -f 2`
    wav=$base$line
    echo $person $profile_id $wav
    #res=` python ./ms-code/Identification/EnrollProfile.py $subscription_key $profile_id $wav true`
    res=`python ./Verification/VerifyFile.py $subscription_key $wav $profile_id`
    #echo $res
    #echo $peron $profile_id $wav >> $log_file
    echo $wav $res $profile_id >> $log_file
    #res=`python ms-code/Identification/CreateProfile.py $subscription_key`
    #for ((i=1;i<length;++i))
    #do
    #    wav='.'${line_list[i]}
    #    echo $person $profile_id $wav
    #    res=` python ./ms-code/Identification/EnrollProfile.py $subscription_key $profile_id $wav true`
    #    echo $res >> $log_file
    #    sleep 5s
    #done
    sleep 2s
done


#for line in $list
#do
#    profile_id=`echo $line | cut -d '/' -f 1`
#    person=`echo $line | cut -d '/' -f 2`
#    echo $profile_id $person
##    #use first 6 wav as enrollments
#    wavs=`ls $base'/'$person`
#    wavs=($wavs)
#    for ((i=0;i<6;++i))
#    do
#        wav=${wavs[$i]}
#        echo $wav
#        res=` python ./ms-code/Identification/EnrollProfile.py $subscription_key $profile_id $base'/'$person'/'$wav true`
#        echo $res >> $log_file
#        echo $res 
#        sleep 5s
#    done
#    sleep 5s
#    #break
#done
