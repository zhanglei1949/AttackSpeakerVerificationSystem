# This directory contains the code for whitebox attack, basically cw attack

Start from a source wav, by adding some noise, make it classified as a target speaker, with sematic content not changed. 

## Design
The network only accept input of shape (?, 25840, 1), corresponding to a feature vector of shape (?, 160, 64, 1).

So, should we cut the source wav into clips, and attack for each clip, or attack segments by segments? Currenly we do the clip first.

And using hearing threshold to limit the noise added

## requirement
python >= 3.5
tensorflow <= 1.4
keras <= 2.1
soundfile 
librosa

