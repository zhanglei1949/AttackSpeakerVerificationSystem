wav_dir='/home/lei/2019/dataset/LibriSpeech/dev-clean-clip/'
adv_dir='./adv/'
source_files=('84-121123-0001-clip.wav' '174-50561-0006-clip.wav' '251-118436-0023-clip.wav' '422-122949-0022-clip.wav' '652-129742-0018-clip.wav')
length=${#source_files[@]}

for ((i = 0; i < $length; ++i))
do
    source=${source_files[$i]}
    target=${source_files[(((i+1)%length))]}
    adv_path=`echo $source | cut -d '.' -f 1`
    target_id=`echo $target | cut -d '-' -f 1`
    adv_path=$adv_path"-"$target_id".wav"
    command="python black-box-attack.py --source_audio $wav_dir$source --target_audio $wav_dir$target --num_steps 500 --adv_audio_path $adv_dir$adv_path"
    echo $command
    #`$command` >> output.log 2>&1
done
#python black-box-attack.py --source_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/84-121123-0001-clip.wav --target_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/174-50561-0006-clip.wav --num_steps 500 --adv_audio_path ./adv/84-121123-0001-clip-174.wav
#python black-box-attack.py --source_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/174-50561-0006-clip.wav --target_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/251-118436-0023-clip.wav --num_steps 500 --adv_audio_path ./adv/174-50561-0006-clip-251.wav
#python black-box-attack.py --source_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/251-118436-0023-clip.wav --target_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/422-122949-0022-clip.wav --num_steps 500 --adv_audio_path ./adv/251-118436-0023-clip-422.wav
#python black-box-attack.py --source_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/422-122949-0022-clip.wav --target_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/652-129742-0018-clip.wav --num_steps 500 --adv_audio_path ./adv/422-122949-0022-clip-652.wav
#python black-box-attack.py --source_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/652-129742-0018-clip.wav --target_audio /home/lei/2019/dataset/LibriSpeech/dev-clean-clip/84-121123-0001-clip.wav --num_steps 500 --adv_audio_path ./adv/652-129742-0018-clip-84.wav