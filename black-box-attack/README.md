# Overview
This part present the black-box attack agains a speaker verfication system.

## Target Model
The victim model is a kaldi model provided by David Snyder, and is based on i-vector and x-vector.

## How to attack
Basically, we use gradient estimation to estimate the gradient and add perturbations to obtain the attack example.