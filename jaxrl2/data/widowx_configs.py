import os

DATA_PATH = '/home/asap7772/kun2/binsort_bridge_1109/11_09_collect_multitask_fulltrajlen/'
ALL_DATASETS = os.listdir(DATA_PATH)

def debug_config():
    return ["/home/asap7772/binsort_bridge/test/actionnoise0.0_binnoise0.0_policysorting_sparse0/train/out.npy"]

def sorting_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS if 'sorting' in x]

def sorting_nobinnoise_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS if 'sorting' in x and 'binnoise0.0' in x]

def sorting_nonzerobinnoise_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS if 'sorting' in x and 'binnoise0.0' not in x]

def pickplace_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS if 'pickplace' in x]

def sorting_pickplace_dataset(train=True):
    suffix = 'train' if train else 'test'
    suffix = "/" + suffix + "/" + 'out.npy'
    return [DATA_PATH + x + suffix for x in ALL_DATASETS]

def check_datasets():
    curr_datasets=dict(
        sorting=sorting_dataset(),
        sorting_nobinnoise_dataset=sorting_nobinnoise_dataset(),
        sorting_nonzerobinnoise_dataset=sorting_nonzerobinnoise_dataset(),
        sorting_pickplace=sorting_pickplace_dataset(),
        pickplace=pickplace_dataset(),
    )
    for name, lst in curr_datasets.items():
        print(name, len(lst))
        for x in lst:
            assert os.path.exists(x), x
            print(x)
        print()

    print("all good")
    
if __name__ == '__main__':
   check_datasets() 