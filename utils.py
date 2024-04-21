import json
import argparse

import numpy as np
import flair
import torch

from flair.data import Sentence
from pycorenlp import StanfordCoreNLP



class TextPreprocessor:

    def __init__(self, nlp=None, properties=None):
        self.nlp = StanfordCoreNLP('http://localhost:9000') if nlp is None else nlp
        self.properties = {'annotators': 'ssplit', 'outputFormat': 'json'} if properties is None else properties

    def sentence_split(self, text, properties={'annotators': 'ssplit', 'outputFormat': 'json'}):
        """Split sentence using Stanford NLP"""

        annotated_string = self.nlp.annotate(text, properties)
        annotated = json.loads(annotated_string)

        sentence_split = list()
        for sentence in annotated['sentences']:
            s = [t['word'] for t in sentence['tokens']]
            k = [item.lower() for item in s if item not in [",", ".", '...', '..']]
            sentence_split.append(" ".join(k))

        return sentence_split

    def preprocess(self, text):
        text = text.replace("\'\'", "").replace(".", ". ")
        sentences = self.sentence_split(text)
        
        results = []
        for sentence in sentences:
            input_str = sentence.lower()
            input_str = input_str.strip()
            input_str = input_str.replace("," , " ,").replace("-rrb-", ")").replace("-lrb-", "(")
            input_str = input_str.split()
            punc = '!"#$%&*+,/:;<=>?@[\]^_`{|}~'
            table = str.maketrans('', '', punc)
            stripped = [w.translate(table) for w in input_str]
            stripped = [w for w in stripped if w]
            results.append(" ".join(stripped))
        
        return results

class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, task='textual_cloze', file='data/recipeqa-train.json', cuda_option=0):
        self.task = task
        
        with open(file, 'r') as myfile:
            data = myfile.read()
        data = json.loads(data)
        self.data = [item for item in data['data'] if item['task'] == task]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
    
def embedding(text, embedder):
    sentence = Sentence(text)
    embedder.embed(sentence)
    if not sentence:
        return torch.zeros(2048)
    return torch.stack([w.embedding for w in sentence])

def prepare_language(text, embedder, cuda_option):
    data = embedding(text, embedder)
    data = data.unsqueeze(0).to(device=cuda_option)
    return data

def read_data(file:str):

    with open(file, 'r') as myfile:
        data = myfile.read()
    
    info = json.loads(data)
    
    visual_coherence = [data for data in info['data'] if data['task']=="visual_coherence"]
    textual_cloze = [data for data in info['data'] if data['task']=="textual_cloze"]
    visual_ordering = [data for data in info['data'] if data['task']=="visual_ordering"]
    visual_cloze = [data for data in info['data'] if data['task']=="visual_cloze"]
    
    print("size of task textual_cloze:\t" + str(len(textual_cloze)))
    print("size of task visual_cloze:\t" + str(len(visual_cloze)))
    print("size of task visual_coherence:\t" + str(len(visual_coherence)))
    print("size of task visual_ordering:\t" + str(len(visual_ordering)))
    print("size of whole set:\t" + str(len(info['data'])))

    return info, visual_cloze, visual_coherence, visual_ordering, textual_cloze

def prepare_data(_set="train", files=["data/recipeqa-train.json", "data/recipeqa-val.json", "data/recipeqa-test.json"]):
    
    train_file = files[0]
    valid_file = files[1]
    test_file = files[2]

    if _set == "train":
        train, train_visual_cloze, train_visual_coherence, train_visual_ordering, train_textual_cloze = read_data(file=train_file)
        return train, train_visual_cloze, train_visual_coherence, train_visual_ordering, train_textual_cloze
    elif _set == "valid":
        valid, val_visual_cloze, val_visual_coherence, val_visual_ordering, val_textual_cloze = read_data(file=valid_file)
        return valid, val_visual_cloze, val_visual_coherence, val_visual_ordering, val_textual_cloze
    elif _set == "test":
        test, test_visual_cloze, test_visual_coherence, test_visual_ordering, test_textual_cloze = read_data(file=test_file)
        return test, test_visual_cloze, test_visual_coherence, test_visual_ordering, test_textual_cloze


def parse_arguments(mode="train", number=200, _set="train", load=False, iteration=1, cuda=0, path="saves/", log="saves/log.txt", architecture=1, embedding_type=1, loss_mode="all", learning_rate=0.1, score_mode="max", max_pool=True): 
    
    parser = argparse.ArgumentParser(description='Getting the arguments passed')
    
    parser.add_argument('-m','--mode', help='The mode of program',required=False)
    parser.add_argument('-n','--number',help='Number of examples', type=int, required=False)
    parser.add_argument('-i','--iteration',help='Number of iterations', type=int, required=False)
    parser.add_argument('-s','--set',help='Working on which set', required=False)
    parser.add_argument('-l','--load',help='Load or not', type=bool, required=False, default=False)
    parser.add_argument('-c', '--cuda', help='Cuda option', type=int, required=False)
    parser.add_argument('-p', '--path', help='Save and Load path', required=False)
    parser.add_argument('-f', '--file', help='Log file name', required=False)
    parser.add_argument('-a', '--architecture', help='Specify Architecture', type=int, required=False)
    parser.add_argument('-e', '--embedding', help='Embedding', type=int, required=False)
    parser.add_argument('-o', '--loss', help='Loss mode', required=False)
    parser.add_argument('-r', '--rate', help='Learning rate', type=float, required=False)
    parser.add_argument('-y', '--score', help='Aggregating Scores Mode', type=str, required=False)
    parser.add_argument('-x', '--maxpool', help='Using customized maxpool', type=bool, required=False, default=True)
    
    args = parser.parse_args()
    
    if args.mode and args.mode in ["train", "test"]:
        mode = args.mode
    if args.number:
        number = args.number
    if args.set and args.set in ["test", "train", "valid"]:
        _set = args.set
    load = args.load
    if args.iteration:
        iteration = args.iteration
    if args.cuda in [0, 1, 2, 3, 4, 5, 6]:
        cuda = args.cuda
    elif args.cuda and args.cuda == -1:
        cuda = -1
    if args.path:
        path = args.path
    if args.file:
        log = args.file
    if args.architecture:
        architecture = args.architecture
    if args.embedding:
        embedding_type = args.embedding
    if args.loss and args.loss in ["random", "all", "one"]:
        loss_mode = args.loss
    if args.rate:
        learning_rate = args.rate
    if args.maxpool:
        max_pool = args.maxpool
    if args.score and args.score in ["max", "mean"]:
        score_mode = args.score

    return mode, number, _set, load, iteration, cuda, path, log, architecture, embedding_type, loss_mode, learning_rate, score_mode, max_pool, args
