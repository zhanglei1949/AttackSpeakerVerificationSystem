# Should we include the VAD step? nope !
# Insert differentiable fbank extraction before input to network
# need to consider : audio length? audio clip? noise evaluation

#Attack schema design: How to set the threshold
# Using create_test_data in test_model.py to generate (1A, 1P, 100N)
# Select one negative audio as source audio
# first evaluate to get the optimum eer and threshold, then adding noise to surpass this threshold
import numpy as np
import librosa
import keras
import tensorflow as tf
import constants as c
import keras.backend as K
import pandas as pd
import soundfile
import glob
from models import my_convolutional_model
from eval_metrics import evaluate
#from test_model import create_test_data
from attack_utils import cosineDistanceLoss, cal_snr
class Attack():
    def __init__(self, checkpoint_path, step_size = 0.01, num_steps = 200, output_dir = './adversarial'):
        self.step_size = step_size
        self.num_steps = num_steps 
        self.checkpoint_path = checkpoint_path
        self.optimizer =  keras.optimizers.Adam(lr = step_size)
        self.print_steps = self.num_steps / 20
        self.output_dir = output_dir
        # Load model in intialization
        #3. Load model
        input_shape = (25840, )
        model = my_convolutional_model(input_shape, batch_size = 1, num_frames = 160)
        #4. Load layers weigths by name
        model.load_weights(self.checkpoint_path, by_name = True)

        for layer in model.layers[3:]:
            layer.trainable = False
        # Ensure trainable variables
        model.layers[1].trainable = True
        my_loss = cosineDistanceLoss() 
        model.compile(optimizer = self.optimizer, loss = my_loss, metrics= ['accuracy'])
        self.model = model
    def load_wav(self, dataset_dir):
        vox = pd.DataFrame()
        vox['wav'] = glob.glob(dataset_dir + '*.wav')
        vox['speaker_id'] = vox['wav'].apply(lambda x : x.split('/')[-1].split('_')[0])
        num_speakers = len(vox['speaker_id'].unique())
        print("Load {} wavs from {} speakers".format(str(len(vox)), str(num_speakers)))
        return vox
    def calculate_sim(self, target_speaker_model, positive_embedding, neg_embeddings):
        anchor = np.tile(target_speaker_model, (neg_embeddings.shape[0] + 1, 1))
        pos_neg_embeddings = np.concatenate((positive_embedding, neg_embeddings), axis = 0)
        mul = np.multiply(anchor, pos_neg_embeddings)
        sim = np.sum(mul, axis = 1)
        #print(sim)
        return sim

    def attack(self, dataset_dir = c.TEST_WAV_DIR, num_anc = 5, num_pos = 1, num_neg = 30):
        # The goal of this function is to extract several batches of samples like (5AN, 1P, 30N)
        # Enroll with 5AN
        # Build threshold using (embedding, 1P, 29N)
        # Some noise is added to the last negative utterance for attack

        vox = self.load_wav(dataset_dir)
        unique_speakers = list(vox['speaker_id'].unique())
        np.random.shuffle(unique_speakers)
        num_triplets = len(unique_speakers) #40
        success_cnt = 0 
        enrolled_meta = "attack.enroll.meta.txt"
        enrolled_meta_file = open(enrolled_meta, 'w')
        for ii in range(num_triplets):
            print("attack for speaker {}".format(unique_speakers[ii]))
            anchor_positive_file = vox[vox['speaker_id'] == unique_speakers[ii]]
            if len(anchor_positive_file) < 6:
                continue
            anchor_positive_file = anchor_positive_file.sample(n = 6, replace=False)
            anchor_files = anchor_positive_file[:5]
            positive_files = anchor_positive_file[5:6]

            negative_files = vox[vox['speaker_id'] != unique_speakers[ii]].sample(n=num_neg, replace=False)
            #print(len(negative_files['speaker_id'].unique()))
            # Now lanch attack
            source_wav = negative_files[-1:]['wav'].values[0]
            source_speaker_id = negative_files[-1:]['speaker_id'].values[0]
            target_speaker_id = unique_speakers[ii]
        
            path_to_save = self.output_dir + source_wav.split('/')[-1][:-4] + "*" + target_speaker_id + ".wav"
            print("Source {} target {} source_wav {}".format(source_speaker_id, target_speaker_id, source_wav))
            
            enrolled_meta_file.write(path_to_save+'\n')

            # get the embedding vector by averaging 5 embeddings
            # Predict on batch is not supported
            target_speaker_embeddings = []
            for i in range(num_anc):
                enrolled_meta_file.write(anchor_files[i:i+1]['wav'].values[0] + '\n')
                audio,fs = librosa.load(anchor_files[i:i+1]['wav'].values[0], c.SAMPLE_RATE)
                audio = np.reshape(audio, (1, 25840))
                embedding = self.model.predict_on_batch(audio)
                embedding = np.reshape(embedding, (1, 512))
                target_speaker_embeddings.append(embedding)
            target_speaker_model = np.mean(target_speaker_embeddings, axis = 0)
            target_speaker_model = np.reshape(target_speaker_model,(1, 512))
            assert(target_speaker_model.shape == (1, 512)) 
            #get the threshold
            positive_audio,fs = librosa.load(positive_files[0:1]['wav'].values[0], c.SAMPLE_RATE)
            positive_audio = np.reshape(positive_audio, (1,25840))
            positive_embedding = self.model.predict_on_batch(positive_audio)
            neg_embeddings = []
            for i in range(num_neg - 1):
                audio,fs = librosa.load(negative_files[i:i+1]['wav'].values[0], c.SAMPLE_RATE)
                audio = np.reshape(audio, (1, 25840))
                embedding = self.model.predict_on_batch(audio)
                embedding = np.reshape(embedding, (1, 512))
                neg_embeddings.append(embedding)
            neg_embeddings = np.asarray(neg_embeddings)
            neg_embeddings = np.reshape(neg_embeddings, (num_neg - 1, 512))
            # calculate similarity between anchor and positive, negative embeddings
            y_pred = self.calculate_sim(target_speaker_model, positive_embedding, neg_embeddings)
            y_true = np.hstack(([1] * num_pos, np.zeros( num_neg - 1)))
            fm, tpr, acc, eer, target_threshold = evaluate(y_pred, y_true)
            print("fm {} tpr {} acc{} eer [] threshold {}".format(fm, tpr, acc, eer, target_threshold))

            success_cnt += self.attack_simple(source_wav, path_to_save, target_speaker_model, target_threshold)
            #break
        print("success rate", success_cnt, 40)
    def attack_simple(self, source_wav, path_to_save, target_embedding_vector, target_threshold):
        success = 0
         #reach success when the target_threshold is crossed.
        #1. First load audio
        original_audio,fs = librosa.load(source_wav, c.SAMPLE_RATE)
        original_audio = np.reshape(original_audio, (1, 25840))
        #print('original audio shape',original_audio.shape)
        #assert(len(original_audio) == 25840)
        assert(original_audio.shape == (1, 25840))
        assert(fs == 16000)

        #2. Load target embeddings
        target_embedding_vector = np.reshape(target_embedding_vector, (1, 512))
        #print("target vector shape", target_embedding_vector.shape)

        #print(self.model.trainable_weights)
        perturbation = np.zeros((1, 25840))
        zero_weights = np.zeros((25840,))
        self.model.layers[1].set_weights([zero_weights])
        new_audio = original_audio + perturbation
        for i in range(self.num_steps):
            loss,_ = self.model.train_on_batch(new_audio, target_embedding_vector)

            
            perturbation += self.step_size * np.sign(self.model.layers[1].get_weights()[0])
            self.model.layers[1].set_weights([zero_weights])
            new_audio = original_audio + perturbation
            
            if (loss < 1 - target_threshold):
                print("success! at step", i)
                success = 1
                break
            if (i % self.print_steps == 0): 
                print("step ", "loss", i, loss, "perturbation ",  np.max(perturbation), np.min(perturbation), np.mean(perturbation))
        
        # save the adversarial audio
        adversarial_audio = np.reshape(new_audio, (25840,))
        noise = np.reshape(perturbation, (25840,))
        #cal snr
        snr = cal_snr(original_audio, noise)
        print('snr', snr)
        soundfile.write(path_to_save, adversarial_audio, c.SAMPLE_RATE, subtype='PCM_16')
        print("save to", path_to_save)
        return success
if __name__ == '__main__':
    checkpoint_path = './best_checkpoint/best_model60800_0.08824.h5'
    output_dir = './adversarial/'
    attack = Attack(checkpoint_path, 0.0001, 200, output_dir)
    source_wav = '../dataset/vox-test-wav-vad/id10270_5r0dWxy17C8_00001.wav'
    target_embedding_vector_path = './embeddings/id10275.npy'
    #path_to_save = './adversarial/' + source_wav.split('/')[-1][:-4] + '_' + target_embedding_vector_path.split('/')[-1][:-4] + '.wav'
    path_to_save = './adversarial/1.wav'
    #attack.attack(source_wav, path_to_save, target_embedding_vector_path)
    attack.attack()
    '''
    a = np.zeros([1,512])
    b = np.ones([1,512])
    sess =tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    print(sess.run(cosineDistance(a,b)))
    '''
