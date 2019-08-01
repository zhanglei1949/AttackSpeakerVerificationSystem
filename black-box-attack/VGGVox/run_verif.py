import matlab
import matlab.engine

if __name__ == "__main__":
    engine = matlab.engine.start_matlab()
    wav1 = './testfiles/verif/8jEAjG6SegY_0000008.wav'
    wav2 = './testfiles/verif/x6uYqmx31kE_0000001.wav'
    arr = [wav1,wav1]
    for i in range(4):
        arr.append(wav2)
    dists = engine.demo_vggvox_verif_voxceleb2_batch(arr, nargout=1)
    print(type(dists), dists)
    pass