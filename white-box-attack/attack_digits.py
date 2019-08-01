# Should we include the VAD step? nope !
# Insert differentiable fbank extraction before input to network
# need to consider : audio length? audio clip? noise evaluation

#Attack schema design: How to set the threshold
# Using create_test_data in test_model.py to generate (1A, 1P, 100N)
# Select one negative audio as source audio
# first evaluate to get the optimum eer and threshold, then adding noise to surpass this threshold
import matlab
import matlab.engine
import numpy as np
import librosa
import keras
import sys
sys.path.append('../speakerVerificationSystem/')
import tensorflow as tf
import constants as c
import keras.backend as K
import pandas as pd
import soundfile
import glob

from models import my_convolutional_model
from eval_metrics import evaluate
#from test_model import create_test_data
from attack_utils import cosineDistanceLoss, cal_snr, load_wavs, cal_audiospec

DEBUG = 0
class Attack():
    def __init__(self, checkpoint_path, step_size = 0.01, num_steps = 200, wav_dir = c.ATTACK_WAV_DIR, 
    output_dir = './adversarial/', embedding_dir = '../data/embeddings/', log_file = './output.log'):
        self.step_size = step_size
        self.num_steps = num_steps 
        self.checkpoint_path = checkpoint_path
        self.optimizer =  keras.optimizers.Adam(lr = step_size)
        self.print_steps = self.num_steps / 20
        self.output_dir = output_dir
        self.dataset_dir = wav_dir
        self.embedding_dir = embedding_dir
        self.log_file = log_file
        # Load model in intialization
        
        input_shape = (25840, )
        #IF debug == 1
        #self.threshold_plh = K.backend.placeholder(shape=(1, 160, 257), name='threshold_plh')
        threshold_input_shape = ( 160, 257)
        # The spectrogram's shape of orignal audio
        ori_spec_shape = (160, 257)
        model = my_convolutional_model(input_shape, threshold_input_shape, ori_spec_shape, batch_size = 1, num_frames = 160)
        #4. Load layers weigths by name
        model.load_weights(self.checkpoint_path, by_name = True)

        for layer in model.layers[3:]:
            layer.trainable = False
        
        # Ensure trainable variables
        model.layers[1].trainable = True
        model.layers[3].trainable = True
#        print(model.layers)
        my_loss = cosineDistanceLoss() 
        
        model.compile(optimizer = self.optimizer, loss = my_loss, metrics= ['accuracy'])
        self.model = model
    
    
    def calculate_sim(self, target_speaker_model, positive_embedding, neg_embeddings):
        anchor = np.tile(target_speaker_model, (neg_embeddings.shape[0] + 1, 1))
        pos_neg_embeddings = np.concatenate((positive_embedding, neg_embeddings), axis = 0)
        mul = np.multiply(anchor, pos_neg_embeddings)
        sim = np.sum(mul, axis = 1)
        #print(sim)
        return sim

    def find_threshold_for_target(self,  target_speaker, num_pos = 1, num_neg = 30):
        # The goal of this function is to extract several batches of samples like (5AN, 1P, 30N)
        # Enroll with 5AN
        # Build threshold using (embedding, 1P, 29N)
        # Some noise is added to the last negative utterance for attack
        dataset_dir = self.dataset_dir
        vox = load_wavs(dataset_dir)
        unique_speakers = sorted(list(vox['speaker_id'].unique()))
        np.random.shuffle(unique_speakers)

        #print("attack for speaker {}".format(unique_speakers[ii]))
        positive_files = vox[vox['speaker_id'] == target_speaker]

        #We need at least num_anc + num_pos audios from on speaker
        positive_file = positive_files.sample(num_pos, replace=False)
        negative_files = vox[vox['speaker_id'] != target_speaker].sample(n=num_neg, replace=False)

        _rd_threshold = np.random.rand(1, 160, 257)
        _rd_ori_audiospec = np.random.rand(1, 160, 257)

        target_speaker_model = np.load(self.embedding_dir + target_speaker + '.npy')
        #get the threshold
        positive_audio,fs = librosa.load(positive_files[0:1]['wav'].values[0], c.SAMPLE_RATE)
        positive_audio = np.reshape(positive_audio, (1,25840))
        positive_embedding = self.model.predict_on_batch([positive_audio, _rd_threshold, _rd_ori_audiospec])
        neg_embeddings = []
        for i in range(num_neg):
            audio,fs = librosa.load(negative_files[i:i+1]['wav'].values[0], c.SAMPLE_RATE)
            audio = np.reshape(audio, (1, 25840))
            embedding = self.model.predict_on_batch([audio, _rd_threshold, _rd_ori_audiospec])
            embedding = np.reshape(embedding, (1, 512))
            neg_embeddings.append(embedding)
        neg_embeddings = np.asarray(neg_embeddings)
        neg_embeddings = np.reshape(neg_embeddings, (num_neg, 512))
            # calculate similarity between anchor and positive, negative embeddings
        y_pred = self.calculate_sim(target_speaker_model, positive_embedding, neg_embeddings)
        y_true = np.hstack(([1] * num_pos, np.zeros( num_neg )))
        fm, tpr, acc, eer, threshold = evaluate(y_pred, y_true)
        print("fm {}; tpr {}; acc{}; eer {}; threshold {}".format(fm, tpr, acc, eer, threshold))
        return threshold
            #target_threshold = 0.75
            #success_cnt += self.attack_simple(source_wav, path_to_save, target_speaker_model, target_threshold)
            #break
            #if (DEBUG == 1):
            #    break
        #print("success rate", success_cnt, 40)
    def attack_simple(self, source_speaker, target_speaker, path_to_save, target_threshold, target_phrase):
        f = open(self.log_file, 'a+')
        success = 0
        #source_wav = self.dataset_dir + source_wav

        #0. load target speaker embedding
        target_embedding_vector = np.load(self.embedding_dir + target_speaker + '_digits.npy')
        target_embedding_vector = np.reshape(target_embedding_vector, (1, 512))
        #1. First load audio
        output_wav = np.zeros((1,1))
        noise = np.zeros((1,1))
        for index in range(len(target_phrase)):
            digit = target_phrase[index]
            print(digit)
            source_wav = self.dataset_dir + source_speaker + '/' + source_speaker + '_' + digit + '.wav'
            original_audio,fs = librosa.load(source_wav, c.SAMPLE_RATE)
            assert(fs == 16000)
            if (len(original_audio) < 25840):
                original_audio = np.concatenate(( original_audio, np.random.rand(25840 - len(original_audio, ) )*1e-3))
            original_audio = np.reshape(original_audio, (1, len(original_audio)))
            #print(original_audio.shape[1])
            ori_audiospec = np.random.rand(1,160,257)
            _rd_threshold = np.random.rand(1, 160, 257)
            #perturbation = np.zeros((1, 25840),dtype='float32')
            perturbation = np.random.rand(1, 25840) * 1e-3
            
            zero_weights = np.zeros((25840,))
            self.model.layers[1].set_weights([zero_weights])
            new_audio = original_audio + perturbation
        
            steps_token = self.num_steps
            success = 0
            for i in range(self.num_steps):
                loss,_ = self.model.train_on_batch([new_audio, _rd_threshold, ori_audiospec ], target_embedding_vector)

                perturbation += self.step_size * (self.model.layers[1].get_weights()[0])
                
                self.model.layers[1].set_weights([zero_weights])
                new_audio = original_audio + perturbation
            
            
                if (loss < 1 - target_threshold):
                    print("success! at step {} with loss {}\n".format(i, loss))
                    success = 1
                    steps_token = i
                    break
                if (i % self.print_steps == 0): 
                    print("clip {} step {} loss {}  perturbation [max] {} [min] {} [avg] {}".format(index, i,loss, np.max(perturbation), np.min(perturbation), np.mean(perturbation)))
            f.write(path_to_save + ' ' + str(index) + ' ' + str(steps_token) + ' ' + str(loss) +  ' ' + str(success) + '\n')
            #output_wav.append(new_audio)
            output_wav = np.column_stack((output_wav, new_audio))
            noise = np.column_stack((noise, perturbation))
            #if DEBUG == 1:
            #    break
        # save the adversarial audio
        output_wav = output_wav[:,1:]
        output_wav = np.array(output_wav)
        #print(output_wav)
        noise = noise[:, 1:]
        noise = np.asarray(noise)
        adversarial_audio = np.reshape(output_wav, (8*original_audio.shape[1], 1))
        noise = np.reshape(noise, (8*original_audio.shape[1], 1))
        #cal snr
        snr = cal_snr(original_audio, noise)
        print('snr', snr)
        f.write(str(snr) + '\n')
        #if (DEBUG != 1):
        soundfile.write(path_to_save, adversarial_audio, c.SAMPLE_RATE, subtype='PCM_16')
        print("save to", path_to_save)
        
        f.close()
        return success
if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description = "Attack [target] audio with a [source] audio")
    #parser.add_argument('--target_audio', dest= 'target_audio', type=str, help="target audio name", required=True)
    #parser.add_argument('--source_audio', dest= 'source_audio', type=str, help="source audio name", required=True)
    #parser.add_argument('--adv_audio_path', dest= 'adv_audio_path', type=str, help="output path for adversarial audio", required=True)
    
    #read attack trails from the metadata, in a format [source ,target, adv_path]
    #parser.add_argument('--num_steps', dest= 'num_steps', type=int, default = 100, help="num_steps")
    #args = parser.parse_args()
    #Need to enroll and get the embeddings for the target speakers

    checkpoint_path = '../speakerVerificationSystem/checkpoints/model_17200_0.54980.h5'
    output_dir = '../data/adversarial_digits/'
    log_file = './output_threshold_digits.log'
    attack  = Attack(checkpoint_path, step_size = 0.01, num_steps = 400, wav_dir = '../data/digits/', 
                output_dir = output_dir, embedding_dir = '../data/embeddings_digits/', log_file = log_file)
    meta_file = './trials_digits.txt'
    
    f = open(meta_file)
    for line in f.readlines():
        arr = line.strip().split(' ')
        source_speaker = arr[0]
        #target_phrase = source_audio.split('/')[-1].split('_')[1]
        target_phrase = '38204796'
        target_speaker = arr[1]
        adv_path = output_dir + source_speaker + '_' + target_speaker + '_' + target_phrase + '.wav'
        
        #threshold = attack.find_threshold_for_target(target_speaker)
        threshold = 0.3
        print(source_speaker, target_speaker, adv_path, threshold)
        attack.attack_simple(source_speaker, target_speaker, adv_path, target_threshold = threshold, target_phrase=target_phrase)