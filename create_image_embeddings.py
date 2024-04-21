import json, time, argparse
import numpy as np
import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50, ResNet50_Weights, resnet101, resnet152, ResNet101_Weights, ResNet152_Weights
from PIL import Image

def load(img, model):
    assert img is not None, "Image is None"
    assert model is not None, "Model is None"
    
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    if img.shape[-3] == 1:
        img = img.repeat(1, 3, 1, 1)
    if img.shape[-3] > 3:
        img = img[:, :3, :, :]
    img = img.cuda()
    return model(img)

def get_embedding_shape_from_json(file: str) -> tuple:
    with open(file, 'r') as f:
        image_id_maps = json.load(f)
    shape = (max(image_id_maps.values()) + 1, 2048)
    return shape

def process_images(model, base_img_dir: str, json_file : str, save_file_path: str = 'data/embeddings/images/test_image_embeddings.npy') -> int:
    start_time = time.time()
    with open(json_file, 'r') as f:
        image_id_maps = json.load(f)

    embedding_shape = get_embedding_shape_from_json(json_file)
    embeddings = np.zeros(embedding_shape, dtype=np.float16)

    cnt = 0
    for img_path, img_id in image_id_maps.items():
 
        img = Image.open(base_img_dir + img_path)
        embedding = load(img, model)
        embedding = embedding.squeeze().detach().cpu().numpy()
        embeddings[img_id] = embedding

        if cnt % 500 == 0:
            print(f'Processed {cnt} images; Saving embeddings...')
            np.save(save_file_path, embeddings)k
        cnt += 1

    np.save(save_file_path, embeddings)
    print("Successfully saved all the embeddings to ", save_file_path)
    end_time = time.time()
    print(f'Processing time: {end_time- start_time}')

    return end_time - start_time

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_images_dir', type=str, default='data/images/train/images-qa/')
    parser.add_argument('--train_image_ids_map', type=str, default='data/train_image_id_mapping.json')
    parser.add_argument('--train_embeddings_path', type=str, default='data/embeddings/images/train_image_embeddings.npy')
    parser.add_argument('--test_images_dir', type=str, default='data/images/test/images-qa/')
    parser.add_argument('--test_image_ids_map', type=str, default='data/test_image_id_mapping.json')
    parser.add_argument('--test_embeddings_path', type=str, default='data/embeddings/images/test_image_embeddings.npy')
    parser.add_argument('--weights', type=str, default='50')

    args = parser.parse_args()
    train_images_dir = args.train_images_dir
    train_image_ids_map = args.train_image_ids_map
    train_embeddings_path = args.train_embeddings_path
    test_images_dir = args.test_images_dir
    test_image_ids_map = args.test_image_ids_map
    test_embeddings_path = args.test_embeddings_path

    weights = int(args.weights)

    weights = ResNet50_Weights.DEFAULT if weights == 50 else ResNet101_Weights.DEFAULT if weights == 101 else ResNet152_Weights.DEFAULT
    model = resnet50(weights=weights) if weights == ResNet50_Weights.DEFAULT else resnet101(weights=weights) if weights == ResNet101_Weights.DEFAULT else resnet152(weights=weights)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.cuda()

    embeddings = process_images(base_img_dir=test_images_dir, json_file=test_image_ids_map, save_file_path=test_embeddings_path, model=model)
    embeddings = process_images(base_img_dir=train_images_dir, json_file=train_image_ids_map, save_file_path=train_embeddings_path, model=model)
