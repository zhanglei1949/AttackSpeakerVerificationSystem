import argparse
import librosa
import time, os
import numpy as  np
import soundfile as sf
import matlab
import matlab.engine
class AdamOpt():
    def __init__(self, step_size = 0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon = 10e-8, size = 2):
        self.step_size = step_size
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.size = size
        self.m = [0] * size
        self.v = [0] * size
        self.t = [0] * size
    def step(self, gradient, indexes):
        #return the delta
        # gradients: array of size audio_point_batch_size/2
        # return the deltas of the same size
        # add learning rate decay
        delta = []
        #self.step_size = self.step_size * 0.98
        for ii in range(len(indexes)):
            i = indexes[ii]
            self.t[i] += 1
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * gradient[ii]
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * np.power(gradient[ii],2)
            m_ = self.m[i] / ( 1 - np.power(self.beta_1, self.t[i]))
            v_  = self.v[i] / (1 - np.power(self.beta_2, self.t[i]))
            delta.append(-self.step_size * m_ / (np.sqrt(v_) + self.epsilon))
        return delta

class Attack():
    def __init__(self, optimizer_name = 'adam', batch_size = 1, audio_point_batch_size = 2, max_steps = 500):
        #gpu enabled?
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.audio_point_batch_size =  audio_point_batch_size
        self.max_steps = 500
        self.h = 0.0001 # step size using in etimating the gradient
        self.engine = matlab.engine.start_matlab() # should have startup.m contained in the current directory
    def attack(self, target_audio_name, source_audio_name, adv_audio_path, num_steps):
        #call the run verif to get score for a list like [target audio, source_v1, source_v2, source_v3, ...]
        # where the number of versions is defined by audio_point_batch
        self.num_steps = min(self.max_steps, num_steps)
        self.tmp_dir = './tmp/'
        log_file = './log_' + adv_audio_path.split('/')[-1] + '.log'
        f = open(log_file, 'w')
        f.write(str(num_steps) + ' ' + str(self.audio_point_batch_size) + '\n')
        #1. First load the audio
        #target_audio, target_fs = librosa.load(target_audio_name, 16000)
        source_audio, source_fs = librosa.load(source_audio_name, 16000)
        # 1.1 calculate the sampling vector
        melspec = librosa.feature.melspectrogram(y=source_audio, sr=source_fs, n_mels=128, hop_length=512, n_fft = 2048)
        meleny = np.sum(melspec, axis=1)
        sampling_weight = np.repeat(meleny, 141)
        sampling_weight = np.concatenate((sampling_weight, np.zeros((len(source_audio) - sampling_weight.shape[0]))))
        # normlization
        sampling_weight_sum = np.sum(sampling_weight)
        sampling_weight_normlized = sampling_weight / sampling_weight_sum


        #2. Check whether the desired path exsits, if exists, wait for user to check
        if (os.path.isfile(adv_audio_path)):
            print("Deisired adv audio exists, please ctrl+c to stop this")
            time.sleep(5)
        else:
            dirpath = os.path.dirname(adv_audio_path)
            print(dirpath)
            if (not os.path.isdir):
                os.makedirs(dirpath)
            #not working, pass
        
        #3. Now attack.
        # as matlab files read audio from file, to avoid large data transfering,
        # before test, show the original distance
        res = self.engine.demo_vggvox_verif_voxceleb2_batch([target_audio_name, source_audio_name], nargout = 1)
        ###TODO
        # Now using L2 distance, try cosine similarity?
        print("original", res)
        
        opt = AdamOpt(step_size = 0.01, size = 22580)
        
        #storing audios in array
        #target_audio, target_fs = librosa.load(target_audio_name, 16000)
        #input_arr = [0]*self.audio_point_batch_size
        input_arr = [target_audio_name]
        for i in range(self.audio_point_batch_size):
            input_arr.append(self.tmp_dir + source_audio_name.split('/')[-1] + '_' +str(i) + '_a.wav')
            input_arr.append(self.tmp_dir + source_audio_name.split('/')[-1] + '_' +str(i) + '_b.wav')
        
        indexes = np.random.choice(len(source_audio), self.audio_point_batch_size, replace = False, p = sampling_weight_normlized)
        for i in range(self.num_steps):
            # 3.1 generate random numbers
            if (i % 50 == 0):
                indexes = np.random.choice(len(source_audio), self.audio_point_batch_size, replace = False, p = sampling_weight_normlized)
            noise_1 = np.zeros(source_audio.shape)
            time0 =time.time()
            for j in range(len(indexes)):
                index = indexes[j]
                noise_1[index] += self.h # adding the delta
                #input_arr[2*j + 1] = source_audio + noise_1
                #input_arr[2*j + 2] = source_audio - noise_1
                sf.write(input_arr[2 * j + 1], source_audio + noise_1, 16000, subtype = 'PCM_16')
                sf.write(input_arr[2 * (j+1)], source_audio - noise_1, 16000, subtype = 'PCM_16')
                noise_1[index] -= self.h # adding the delta
            #### Log
            time1 = time.time()
            res = np.log(self.engine.demo_vggvox_verif_voxceleb2_batch(input_arr, nargout = 1))
            time2 = time.time()
            #res = self.engine.demo_vggvox_verif_voxceleb2_batch(input_arr, nargout = 1)
            #may be log is better for small values
            #print(res) # returned value is distance, goad is to miniminze
            # Each two items contains f(x+h_i) and f(x-h_i), now calculate the gradients
            # 3.2 gradient cacluation, in cw way
            gradients = [ (res[j] - res[j+1]) / (2 * self.h) for j in range(0, len(res), 2)]
            print(i, np.mean(res))
            time3 = time.time()
            f.write(str(i) + ' ' + str(np.mean(res)) + '\n')

            # 3.3 apply adam optimzer for the computation of delta
            
            deltas = opt.step(gradients, indexes)
            #print(deltas)
            time4 = time.time()
            print(time4- time3, time3 - time2,time2 - time1,time1- time0)
            # 3.3 apply the changes to the source file
            #3.3 apply to the source file
            for j in range(len(indexes)):
                source_audio[indexes[j]] += deltas[j]
            if (i % 50 == 0):
                sf.write(adv_audio_path, source_audio, 16000, subtype='PCM_16')
        # Finally check
        res = self.engine.demo_vggvox_verif_voxceleb2_batch([target_audio_name, adv_audio_path, source_audio_name])
        f.write(str(res[0]) + ' ' + str(res[1]))
        f.close()
        print(res)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Attack [target] audio with a [source] audio")
    parser.add_argument('--target_audio', dest= 'target_audio', type=str, help="target audio name", required=True)
    parser.add_argument('--source_audio', dest= 'source_audio', type=str, help="source audio name", required=True)
    parser.add_argument('--adv_audio_path', dest= 'adv_audio_path', type=str, help="output path for adversarial audio", required=True)
    parser.add_argument('--num_steps', dest= 'num_steps', type=int, default = 500, help="num_steps")

    args = parser.parse_args()
    print(args)
    
    audio_point_batch_size = 384
    attack = Attack(audio_point_batch_size = audio_point_batch_size)
    attack.attack(args.target_audio, args.source_audio, args.adv_audio_path, args.num_steps)
