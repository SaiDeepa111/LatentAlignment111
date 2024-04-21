import json, gc
import flair
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings as BertEmbeddings
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
from src.models import LSTMFlair, FullyConnected, ResidualFullyConnected
import random
import numpy as np
import operator

from src.bertModel import NoPosLXRTEncoder

from utils import read_data, parse_arguments, TextPreprocessor, embedding, prepare_language, prepare_data

def prepare_answer(texts, embedder, cuda_option):
    data = []
    for text in texts:
        embed = embedding(text[0], embedder)[-1]
        position = torch.zeros(4)
        position[text[1]] = 1
        if is_cuda:
            position = position.cuda(cuda_option)
        if len(embed.shape) == 0:
            embed = torch.zeros(2048).cuda(cuda_option)
        if len(position.shape) == 0:
            position = torch.zeros(4).cuda(cuda_option)
        if is_cuda:
            embed = embed.to(cuda_option)
            position = position.to(cuda_option)
        result = torch.cat((embed, position), 0)
        result = result.squeeze(0)
        data.append(result)
        
    data = torch.stack(data)
    data = data.unsqueeze(0)
    if is_cuda:
        data = data.to(device=cuda_option)
    return data

def save_models(iteration):
    it = iteration
    
    torch.save(LSTM_Answer.state_dict(), save_path + "ANSWERL")
    torch.save(LSTM_Img.state_dict(), save_path + "IMGL")
    torch.save(LSTM_Lang.state_dict(), save_path + "LANGL")
    torch.save(contextTransformer.state_dict(), save_path + "ContextT")
    torch.save(answerTransformer.state_dict(), save_path + "AnswerT")
    torch.save(imageTransformer.state_dict(), save_path + "ImgT")
    torch.save(multicoder.state_dict(), save_path + "MultiCoder")
    torch.save(textTransformer.state_dict(), save_path + "TextT")
    
    if iteration % 3 == 0:
        torch.save(LSTM_Answer.state_dict(), save_path + "step" + str(it) + "_ANSWERL")
        torch.save(LSTM_Img.state_dict(), save_path + "step" + str(it) + "_IMGL")
        torch.save(LSTM_Lang.state_dict(), save_path + "step" + str(it) + "_LANGL")
        torch.save(contextTransformer.state_dict(), save_path + "step" + str(it) + "_ContextT")
        torch.save(answerTransformer.state_dict(), save_path + "step" + str(it) + "_AnswerT")
        torch.save(imageTransformer.state_dict(), save_path + "step" + str(it) + "_ImgT")
        torch.save(multicoder.state_dict(), save_path + "step" + str(it) + "_MultiCoder")
        torch.save(textTransformer.state_dict(), save_path + "step" + str(it) + "_TextT")
    
    print('\n\n')


def execute(_m, _n, _s, _iteration, dataset, base_image_path, log_file, cuda_option, save_path, loss_mode, learning_rate, score_mode, max_pool):
    
    if _m == "train":
        params = list(answerTransformer.parameters()) + list(contextTransformer.parameters()) + list(imageTransformer.parameters())
        params += list(LSTM_Answer.parameters()) + list(LSTM_Img.parameters()) + list(LSTM_Lang.parameters())
        params += list(multicoder.parameters()) + list(textTransformer.parameters())   
        optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
    
    for it in tqdm(range(_iteration)):
        logger = open(log_file, "a+")
        print("-----------------", file=logger)
        logger.write("Start of the iteration "+ str(it) +". \n")
        
        total_loss = 0
        number_true = 0
        p2 = 0
        passed = 0
        
        indices = np.random.randint(0, len(dataset), _n)

        for ind in tqdm(indices):
            print("sample number: ", ind, file=logger)
            sample = dataset[ind]
            print("sample id: ", sample['recipe_id'], file=logger)
            
            if _m == "train":
                LSTM_Answer.zero_grad()
                LSTM_Img.zero_grad()
                LSTM_Lang.zero_grad()
                contextTransformer.zero_grad()
                answerTransformer.zero_grad()
                imageTransformer.zero_grad()

                if architecture == 8:
                    multicoder.zero_grad()
                    textTransformer.zero_grad()
                
            if architecture == 8:
                images_list = [instruction_step['images'] for instruction_step in sample['context']]
                img_data = []
                for info in range(len(images_list)):
                    image_tensor = []
                    check = False
                    for item in images_list[info]:
                        _id = images_id[item]
                        check = True
                        image_tensor.append(torch.from_numpy(images_representation[_id]).float().to(cuda_option))
                    if check:
                        img_data.append(torch.stack(image_tensor))
                    else:
                        img_data.append([])

            instructions = [instruction_step['body'] for instruction_step in sample['context']]

            placeholder = 0
            g_placeholder = 0
            question = []
            
            try:
                for q in range(len(sample['question'])):
                    if sample['question'][q] == "@placeholder":
                        placeholder = 1
                        g_placeholder = q
                        continue
                    question.append([sample['question'][q], q])
                question_result = LSTM_Answer(prepare_answer(texts=question, embedder = selected_embedding, cuda_option = cuda_option))[-1][-1]
            except:
                print(q)
                print(sample['question'])
                raise

            answers = []
            for answer_choice in sample['choice_list']:
                answers.append([answer_choice, g_placeholder])
            correct_answer = [sample['choice_list'][sample['answer']], g_placeholder]
            del answers[sample['answer']]
            
            # Delete Empty Answers
            _list = []
            for _it in range(len(answers)):
                try:
                    if answers[_it][0] == "" or answers[_it][0] == " ":
                        _list.insert(0, _it)
                except:
                    print(_it)
                    raise
            for item in _list:
                del answers[item]


            answers_results = []
            answers_results.append(LSTM_Answer(prepare_answer(texts=[correct_answer], embedder = selected_embedding, cuda_option = cuda_option))[-1][-1])
            
            for answer in answers:
                answers_results.append(LSTM_Answer(prepare_answer(texts=[answer], embedder = selected_embedding, cuda_option = cuda_option))[-1][-1])
            answers_results = torch.stack(answers_results)
            answer_results = answerTransformer(answers_results)

            results = []
            
            try:                
                for _it in range(len(instructions)):
                    
                    sentences = text_preprocessor.preprocess(instructions[_it])
                    all_text = []
                    
                    try:
                        for sentence in sentences:
                            if len(sentence) > 3:
                                _input = prepare_language(text= sentence, embedder = selected_embedding, cuda_option = cuda_option)
                                all_text.append(_input.squeeze(0))
                    except:
                        print(sample)
                        print(sentences)
                        raise
                    if architecture == 8:
                        if not len(all_text):
                            continue
                        all_text = torch.cat(all_text, 0)
                        all_text = textTransformer(all_text)

                        if len(img_data[_it]):
                            all_text, vision_input = multicoder(lang_feats=all_text.unsqueeze(0),
                                                                visn_feats=img_data[_it].unsqueeze(0), 
                                                                visn_attention_mask=None, 
                                                                lang_attention_mask=None)
                            
                            sentences_result = LSTM_Lang(all_text)[-1][-1]
                            image_result = LSTM_Img(vision_input)[-1][-1]
                        else:
                            print("\nNo images for this step")
                            all_text, vision_input = multicoder(lang_feats=all_text.unsqueeze(0),visn_feats=None, visn_attention_mask=None, lang_attention_mask=None)
                            image_result = torch.zeros(2048).cuda(cuda_option)
                            sentences_result = LSTM_Lang(all_text)[-1][-1]

                    value = contextTransformer(torch.cat((sentences_result, question_result, image_result)))
                    
                    results.append(value)

                results = torch.stack(results)
                a_norm =  answer_results/ answer_results.norm(dim=1)[:, None]
                b_norm = results / results.norm(dim=1)[:, None]
                final_results = torch.mm(a_norm, b_norm.transpose(0,1))
#                 print("final results is: ", final_results, file=logger)
                if max_pool:
                    r11 = final_results.clone()
                    indexes = []
                    for i in range(4):
                        v, i = r11.max(1)
                        v1, i1 = v.flatten().max(0)
                        indexes.append((i1.item(), i[i1.item()].item()))
                        j = torch.arange(r11.size(0)).long()
                        r11[j, i[i1.item()].item()] = -100000000000
                        r11[i1.item()][:] = -100000000000
                    indexes.sort(key = operator.itemgetter(0))
                    index0 = indexes[0][1]
                    index1 = []
                    index2 = []
                    for item in indexes:
                        index1.append(item[0])
                        index2.append(item[1])
                    results = final_results[index1, index2]
                else:
                    results, indexes = final_results.max(1)
                    index0 = indexes[0]
#                 print("the matching maxes are: ", results, file=logger)
                most, index_most = torch.max(results,0)
                print_results = {}
                print_results[sample['answer']] = results[0].item()
                for tt in range(4):
                    if tt == sample['answer']:
                        continue
                    if tt < sample['answer']:
                        print_results[tt] = results[tt+1].item()
                    else:
                        print_results[tt] = results[tt].item()
                print_results_list = []
                for tt in range(4):
                    print_results_list.append(print_results[tt])
                
                print("the matching result is: ", print_results_list, file=logger)
                print("the predicted answer: ", np.argmax(print_results_list), file=logger)
                print("The answer is: ", sample['answer'], file=logger)
                if sample['answer'] == np.argmax(print_results_list):
                    print("correct Answer", file=logger)
                else:
                    print("wrong Answer", file=logger)

                checking_p2 = torch.tensor(print_results_list).topk(2)[1]
                if sample['answer'] in checking_p2:
                    p2 += 1
                if index_most == 0:
                    number_true += 1
                #     print("correct number: ", number_true, file=logger)

                if _m == "train":
                    if loss_mode == "one":
                        keys = [1, 2, 3]
                        keys = [key for key in keys if key < final_results.shape[0]]
                        ri = random.choice(keys)
                        loss = 0
                        for key in keys:
                            loss += max(0, final_results[key][index0] - results[0] + 0.1)
                        for ind in range(final_results[ri].shape[0]):
                            loss += max(0, results[ri] - results[0] + 0.1)
                    else:
                        loss = 1- results[0]
                        keys = [1, 2, 3]
                        keys = [key for key in keys if key < final_results.shape[0]]
                        for _it in keys:
                            loss += max(0, results[_it] - 0.1)

                    if loss != 0:
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()

            except KeyboardInterrupt:
                logger.close()
                raise
            except:
                raise
                print("GPU PASS")
                passed += 1
                    
        if _m == "train":
            print("total loss is" , total_loss)
            logger.write("The loss of this iteration is "+ str(total_loss) +". \n")
            save_models(iteration=_it)

        print("The ratio of being correct is: ", number_true / (_n - passed))
        print("The ratio of p2 correct is: ", p2 / (_n - passed), file=logger)
        logger.write("The accuracy is "+ str((number_true / (_n - passed))) +". \n")
        logger.close()
        

def main(mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, loss_mode, learning_rate, score_mode, max_pool, args):
    
    save_path = str(save_path)
    
    logger = open(log_file,"a+")
    logger.write("\n --------------- \n Start of the model execution. \n")
    logger.close()
    
    with open(log_file, 'a+') as f:
        json.dump(args.__dict__, f, indent=2)
    logger = open(log_file,"a+")
    logger.write("\n start the training \n")
    logger.close()

    #prepare the data
    _, _, _, _, data_tc = prepare_data(_set)

    LSTM_Lang.cuda(cuda_option)
    LSTM_Img.cuda(cuda_option)
    LSTM_Answer.cuda(cuda_option)
    contextTransformer.cuda(cuda_option)
    answerTransformer.cuda(cuda_option)
    imageTransformer.cuda(cuda_option)
    flair.device = torch.device(cuda_option) 
    multicoder.cuda(cuda_option)
    textTransformer.cuda(cuda_option)

    #check for the loading parameters
    if load:
        print("Loading the existing parameters of the models")
        
        LSTM_Lang.load_state_dict(torch.load(save_path + "LANGL"))
        LSTM_Img.load_state_dict(torch.load(save_path + "IMGL"))
        LSTM_Answer.load_state_dict(torch.load(save_path + "ANSWERL"))
        contextTransformer.load_state_dict(torch.load(save_path + "ContextT"))
        answerTransformer.load_state_dict(torch.load(save_path + "AnswerT"))
        imageTransformer.load_state_dict(torch.load(save_path + "ImgT"))

        multicoder.load_state_dict(torch.load(save_path + "MultiCoder"))
        textTransformer.load_state_dict(torch.load(save_path + "TextT"))
    
    #set to training
    if mode == "train" and load:
        LSTM_Lang.train()
        LSTM_Img.train()
        LSTM_Answer.train()
        contextTransformer.train()
        answerTransformer.train()
        imageTransformer.train()
        multicoder.train()
        textTransformer.train()
    elif mode == "test" and load:
        LSTM_Lang.eval()
        LSTM_Img.eval()
        LSTM_Answer.eval()
        contextTransformer.eval()
        answerTransformer.eval()
        imageTransformer.eval()
        multicoder.eval()
        textTransformer.eval()

    if _set == "train":
        base_image_path = 'images-qa/train/images-qa/'
    elif _set == "test":
        base_image_path = 'images-qa/test/images-qa/'
    elif _set == "valid":
        base_image_path = 'images-qa/val/images-qa/'

    execute(mode, number, _set, iteration, data_tc, base_image_path, log_file, cuda_option, save_path, loss_mode, learning_rate, score_mode, max_pool)


if __name__ == "__main__":

    mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, embedding_type, loss_mode, learning_rate, score_mode, max_pool, args = parse_arguments()

    text_preprocessor = TextPreprocessor()

    selected_embedding = FlairEmbeddings("news-forward")
    embed_dim = 2048

    flair = FlairEmbeddings("news-forward")

    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-1]
    resnet = nn.Sequential(*modules)
    resnet.eval()

    scaler = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    is_cuda = torch.cuda.is_available()

    LSTM_Lang = LSTMFlair(input_dim=embed_dim, hidden_dim=2048, batch_size = 1) 
    LSTM_Img = LSTMFlair(input_dim=2048, hidden_dim=2048, batch_size = 1)
    LSTM_Answer = LSTMFlair(input_dim=embed_dim+4, hidden_dim=2048, batch_size = 1)  

    multicoder = NoPosLXRTEncoder(visual_feat_dim=2048, drop=0.0, l_layers=3, x_layers=2, r_layers=1, num_attention_heads=4, hidden_size=2048, intermediate_size=2048)
    LSTM_Lang = LSTMFlair(input_dim=2048, hidden_dim=2048, batch_size = 1)
    textTransformer = FullyConnected(dims = [embed_dim, 2048, 2048], layers = 2)
    contextTransformer = ResidualFullyConnected(dims = [6144, 2048, 1024, 512, 512], layers = 4)
    answerTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    imageTransformer = ResidualFullyConnected(dims = [2048, 1024, 1024, 512, 512], layers = 4)
    
    train_image_ids_map = 'data/train_image_id_mapping.json'
    test_image_ids_map = 'data/test_image_id_mapping.json'
    train_embeddings_load_path = 'data/embeddings/images/train_image_embeddings_resnet50.npy'
    test_embeddings_load_path = 'data/embeddings/images/test_image_embeddings_resnet50.npy'
    
    if _set == "train":
        
        images_representation = np.load(train_embeddings_load_path, allow_pickle=True).astype(np.float16)
        print("Successfully loaded train image embeddings with shape ", images_representation.shape)
        
        with open(train_image_ids_map, 'r') as f:
            images_id = json.load(f)
            print("Successfully loaded train image ids with length ", len(images_id))
    
    elif _set == "test":
        
        images_representation = np.load(test_embeddings_load_path, allow_pickle=True).astype(np.float16)
        print("Successfully loaded test image embeddings with shape ", images_representation.shape)
        
        with open(test_image_ids_map, 'r') as f:
            images_id = json.load(f)
            print("Successfully loaded test image ids with length ", len(images_id))

    main(mode, number, _set, load, iteration, cuda_option, save_path, log_file, architecture, loss_mode, learning_rate, score_mode, max_pool, args)
