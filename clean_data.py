
from datasets import load_dataset


ds = load_dataset("slugroom/rjochtwurd-dataset")


# def clean_data(ds):
#     train_data = ds['train']
#     test_data = ds['test']
#
#     # index of duplicate data
#     duplicate_train = []
#
#     # remove duplicates from train and test data
#     for i in range(len(train_data)):
#         txt = train_data[i]['text']
#
#         for j in range(i+1, len(train_data)):
#             if txt == train_data[j]['text']:
#                 duplicate_train.append(j)
#
#
#     train_data = train_data.filter(lambda example, idx: idx not in duplicate_train, with_indices=True)
#     ds['train'] = train_data
#
#     return ds


def clean_data(ds):
    train_data = ds['train']
    test_data = ds['test']
    
    # create a set of unique text examples from the test data
    test_text_set = set([example['prediction'] for example in test_data])
    
    # filter out any train data that is also in the test set
    train_data = train_data.filter(lambda example: example['prediction'] not in test_text_set)
    
    print(f"Removed {len(ds['train']) - len(train_data)} duplicate examples from the training data.")
    ds['train'] = train_data
    
    return ds

ds = clean_data(ds)

ds.push_to_hub("slugroom/rjochtwurd-dataset")
