#!/bin/bash
#create profile ids and store them in a txt
outputfile='profile_person_jinlei'
#subscription_key='874b92dfcd7742baa1c653fa4d3659f5'
#subscription_key='0f3f4ad861274e3ba7e83f5e09dda858'
subscription_key=`cat ./Verification/subscriptionkey`
persons="ph1_ ph2_ ph3_ ph4_ ph5_ ph6_ ph7_ ph8_ ph9_ ph10_"
person_list=($persons)
length=${#person_list[@]}
echo $length
for ((i=0;i<10;++i))
do
    person=${person_list[$i]}
    echo $person
    res=`python Verification/CreateProfile.py $subscription_key `
    profile_id=`echo $res | cut -d ' ' -f 4`
    echo $profile_id $person
    echo $profile_id' '$person >> $outputfile
    #cnt_sa=`ls $base'/'$person'/'sa* | wc -l`

    #cnt_sx=`ls $base'/'$person'/'sx* | wc -l`
    #cnt_si=`ls $base'/'$person'/'si* | wc -l`
    #echo $person $cnt_sa $cnt_sx $cnt_si >> $outputfile
done
