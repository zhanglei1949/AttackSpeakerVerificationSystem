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

DEBUG = 1
class Attack():
    def __init__(self, checkpoint_path, step_size = 0.01, num_steps = 200, wav_dir = c.ATTACK_WAV_DIR, output_dir = './adversarial/', embedding_dir = '../data/embeddings/'):
        self.step_size = step_size
        self.num_steps = num_steps 
        self.checkpoint_path = checkpoint_path
        self.optimizer =  keras.optimizers.Adam(lr = step_size)
        self.print_steps = self.num_steps / 20
        self.output_dir = output_dir
        self.dataset_dir = wav_dir
        self.embedding_dir = embedding_dir
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

    def attack(self,  num_anc = 5, num_pos = 1, num_neg = 30):
        # The goal of this function is to extract several batches of samples like (5AN, 1P, 30N)
        # Enroll with 5AN
        # Build threshold using (embedding, 1P, 29N)
        # Some noise is added to the last negative utterance for attack
        dataset_dir = self.dataset_dir
        vox = load_wavs(dataset_dir)
        if DEBUG == 1:
            unique_speakers = sorted(list(vox['speaker_id'].unique()))
        else: 
            unique_speakers = list(vox['speaker_id'].unique())
            np.random.shuffle(unique_speakers)
        num_triplets = len(unique_speakers) #40
        success_cnt = 0 
        enrolled_meta = "attack.enroll.meta.txt"
        enrolled_meta_file = open(enrolled_meta, 'w')
        # Write the enroll information out to a file
        for ii in range(num_triplets):
            print("attack for speaker {}".format(unique_speakers[ii]))
            anchor_positive_file = vox[vox['speaker_id'] == unique_speakers[ii]]

            #We need at least num_anc + num_pos audios from on speaker
            if len(anchor_positive_file) < num_anc + num_pos:
                continue
            if DEBUG == 1: 
                anchor_positive_file = anchor_positive_file[:num_anc + num_pos]
            else:
                anchor_positive_file = anchor_positive_file.sample(n = num_anc + num_pos, replace=False)
            anchor_files = anchor_positive_file[: num_anc]
            positive_files = anchor_positive_file[num_anc : num_anc + num_pos]

            if DEBUG == 1:
                negative_files = vox[vox['speaker_id'] != unique_speakers[ii]][:num_neg]
            else:
                negative_files = vox[vox['speaker_id'] != unique_speakers[ii]].sample(n=num_neg, replace=False)
            #print(len(negative_files['speaker_id'].unique()))
            # Now lanch attack
            source_wav = negative_files[-1:]['wav'].values[0]
            source_speaker_id = negative_files[-1:]['speaker_id'].values[0]
            target_speaker_id = unique_speakers[ii]
        
            path_to_save = self.output_dir + source_wav.split('/')[-1][:-4] + "-" + target_speaker_id + ".wav"
            print("target speaker {}; source_wav {}".format(target_speaker_id, source_wav))
            enrolled_meta_file.write("attack speaker {}".format(target_speaker_id)) 
            enrolled_meta_file.write(path_to_save+'\n')

            # Randomly generate a mask matrix, not effecting the results
            _rd_threshold = np.random.rand(1, 160, 257)

            # Randomly Generate a spec matrix, not effecting the forwarding results
            _rd_ori_audiospec = np.random.rand(1, 160, 257)
            # get the embedding vector by averaging 5 embeddings
            # Predict on batch is not supported
            target_speaker_embeddings = []
            for i in range(num_anc):
                enrolled_meta_file.write(str(i) + " " + anchor_files[i:i+1]['wav'].values[0] + '\n')
                audio,fs = librosa.load(anchor_files[i:i+1]['wav'].values[0], c.SAMPLE_RATE)
                audio = np.reshape(audio, (1, 25840))
                
                
                embedding = self.model.predict_on_batch([audio, _rd_threshold, _rd_ori_audiospec])
                embedding = np.reshape(embedding, (1, 512))
                target_speaker_embeddings.append(embedding)
            target_speaker_model = np.mean(target_speaker_embeddings, axis = 0)
            target_speaker_model = np.reshape(target_speaker_model,(1, 512))
            assert(target_speaker_model.shape == (1, 512)) 

            # Save the target speaker embedding
            np.save(self.embedding_dir + target_speaker_id + '.npy', target_speaker_model)
            #get the threshold
            positive_audio,fs = librosa.load(positive_files[0:1]['wav'].values[0], c.SAMPLE_RATE)
            positive_audio = np.reshape(positive_audio, (1,25840))
            positive_embedding = self.model.predict_on_batch([positive_audio, _rd_threshold, _rd_ori_audiospec])
            neg_embeddings = []
            for i in range(num_neg - 1):
                audio,fs = librosa.load(negative_files[i:i+1]['wav'].values[0], c.SAMPLE_RATE)
                audio = np.reshape(audio, (1, 25840))
                embedding = self.model.predict_on_batch([audio, _rd_threshold, _rd_ori_audiospec])
                embedding = np.reshape(embedding, (1, 512))
                neg_embeddings.append(embedding)
            neg_embeddings = np.asarray(neg_embeddings)
            neg_embeddings = np.reshape(neg_embeddings, (num_neg - 1, 512))
            # calculate similarity between anchor and positive, negative embeddings
            y_pred = self.calculate_sim(target_speaker_model, positive_embedding, neg_embeddings)
            y_true = np.hstack(([1] * num_pos, np.zeros( num_neg - 1)))
            fm, tpr, acc, eer = evaluate(y_pred, y_true)
            print("fm {}; tpr {}; acc{}; eer {}".format(fm, tpr, acc, eer))
            target_threshold = 0.75
            success_cnt += self.attack_simple(source_wav, path_to_save, target_speaker_model, target_threshold)
            #break
            #if (DEBUG == 1):
            #    break
        print("success rate", success_cnt, 40)
    def attack_simple(self, source_wav, path_to_save, target_embedding_vector, target_threshold):
        success = 0

        #1. First load audio
        original_audio,fs = librosa.load(source_wav, c.SAMPLE_RATE)
        original_audio = np.reshape(original_audio, (1, 25840))
        assert(original_audio.shape == (1, 25840))
        assert(fs == 16000)

        #1.1 Calculate the spectrogram in db
        ori_audiospec = cal_audiospec(original_audio, fs)
        ori_audiospec = np.reshape(ori_audiospec, (1, 160, 257))

        #2. Load target embeddings
        target_embedding_vector = np.reshape(target_embedding_vector, (1, 512))
        #print("target vector shape", target_embedding_vector.shape)
        
        #3. If DEBUG, compute hearing threshold.
        if (DEBUG == 1):
            engine = matlab.engine.start_matlab()
            res1, res2 = engine.test_threshold(source_wav, nargout = 2)
            #res1 = engine.workspace['remapped_heating_thr']
            #res2 = engine.workspace['remapped_heating_thr_dB']
            hearingThreshold  = np.asarray(res1)
            hearingThreshold_dB = np.asarray(res2)
            
            #Hearing threshold is normalized to (0,1)
            # To padding (1, 160, 256) to (1,160, 257)
            #print(hearingThreshold.shape) (161, 256)
            hearingThreshold = np.reshape(hearingThreshold[:160,:], (1, 160, 256))
            hearingThreshold = np.concatenate([hearingThreshold, hearingThreshold[:,:,255:]], axis = 2)
            print("hearing threshold after concatenate", hearingThreshold.shape)

            # supposed to be (1,160, 257)
            hearingThreshold_dB = np.reshape(hearingThreshold_dB[:160, :], (1, 160, 256))
            hearingThreshold_dB = np.concatenate([hearingThreshold_dB, hearingThreshold_dB[:,:,255:]], axis = 2)
            
            #substracting the hearing threshold with original audiospec
            #substraction = hearingThreshold_dB - ori_audiospec
            #print("max ", np.max(substraction), "min ", np.min(substraction), "mean ", np.mean(substraction))


        perturbation = np.zeros((1, 25840))
        zero_weights = np.zeros((25840,))
        self.model.layers[1].set_weights([zero_weights])
        new_audio = original_audio + perturbation
        for i in range(self.num_steps):
            loss,_ = self.model.train_on_batch([new_audio, hearingThreshold, ori_audiospec ], target_embedding_vector)

            perturbation += self.step_size * (self.model.layers[1].get_weights()[0])
            #print(perturbation[0][0])

            #if DEBUG == 1:

                #trainable_tensors = self.model.trainable_weights
                #print(trainable_tensors)
                #gradients = K.gradients(self.model.output, trainable_tensors)
                #print(gradients)
                #sess = K.get_session()
                #_gradients = sess.run(gradients, feed_dict={self.model.input[0] : new_audio, 
                #                                            self.model.input[1] : hearingThreshold,
                #                                            self.model.input[2] : ori_audiospec})
                #sess = tf.Session()
                #print(sess.run(gradients))
                #print(" gradients",_gradients[0][0])
                #print(" weight matrix", self.model.layers[1].get_weights()[0][0])
                
            self.model.layers[1].set_weights([zero_weights])
            new_audio = original_audio + perturbation
            
            
            if (loss < 1 - target_threshold):
                print("success! at step {} with loss {}\n".format(i, loss))
                success = 1
                break
            if (i % self.print_steps == 0): 
                print("step {} loss {}  perturbation [max] {} [min] {} [avg] {}".format(i,loss, np.max(perturbation), np.min(perturbation), np.mean(perturbation)))
            
            #if DEBUG == 1:
            #    break
        # save the adversarial audio
        adversarial_audio = np.reshape(new_audio, (25840,))
        noise = np.reshape(perturbation, (25840,))
        #cal snr
        snr = cal_snr(original_audio, noise)
        print('snr', snr)
        #if (DEBUG != 1):
        soundfile.write(path_to_save, adversarial_audio, c.SAMPLE_RATE, subtype='PCM_16')
        print("save to", path_to_save)
        return success
if __name__ == '__main__':
    checkpoint_path = '../speakerVerificationSystem/checkpoints/model_17200_0.54980.h5'
    output_dir = '../data/adversarial/'
    attack = Attack(checkpoint_path, 0.01, 300,c.ATTACK_WAV_DIR ,output_dir)
    attack.attack()
    '''
    a = np.zeros([1,512])
    b = np.ones([1,512])
    sess =tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    print(sess.run(cosineDistance(a,b)))
    '''
