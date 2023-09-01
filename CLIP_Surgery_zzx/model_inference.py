import torch.nn as nn
from clip_zzx.simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Union, List
from pkg_resources import packaging
import torch



# define text clip and image clip seperately to enable model gpu parallel
class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)


_tokenizer = _Tokenizer()
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
    
def encode_text_with_prompt_ensemble_anomaly(model, category, device):
    
    template_level_prompts = [
        'a cropped photo of the {}',
        'a cropped photo of a {}',
        'a close-up photo of a {}',
        'a close-up photo of the {}',
        'a bright photo of a {}',
        'a bright photo of the {}',
        'a dark photo of the {}',
        'a dark photo of a {}',
        'a jpeg corrupted photo of a {}',
        'a jpeg corrupted photo of the {}',
        'a blurry photo of the {}',
        'a blurry photo of a {}',
        'a photo of a {}',
        'a photo of the {}',
        'a photo of a small {}',
        'a photo of the small {}',
        'a photo of a large {}',
        'a photo of the large {}',
        'a photo of the {} for visual inspection',
        'a photo of a {} for visual inspection',
        'a photo of the {} for anomaly detection',
        'a photo of a {} for anomaly detection'
    ]
    
    state_level_normal_prompts = [
        '{}',
        'flawless {}',
        'perfect {}',
        'unblemished {}',
        '{} without flaw',
        '{} without defect',
        '{} without damage'
        #'three flawless, perfect and unblemished {} with different colors without any defect, flaw or damage in a bowl',
        #    'flawless, perfect and unblemished {} without any defect, flaw or damage'
    ]

    state_level_abnormal_prompts = [
        #   'damaged {} with flaw or defect or damage',
            'damaged {}',
            '{} with flaw',
            '{} with defect',
            '{} with damage',
        ##   '{} with missing parts'
        ##   '{} with print',  # added
        ##    '{} with hole',  # added
        ##   '{} with crack', # added
        ##   '{} with scratch', # added
        ##    '{} with discoloration',
        ##    '{} with stains',
        ##    '{} with missing parts',
        ##    '{} with broken parts',
        ##    '{} with bumpy surfaces'
    ]

    normal_phrases = []
    abnormal_phrases = []

    # some categories can be renamed to generate better embedding
    #if category == 'grid':
    #    category  = 'chain-link fence'
    #if category == 'toothbrush':
    #    category = 'brush' #'brush' #
    for template_prompt in template_level_prompts:
        # normal prompts
        for normal_prompt in state_level_normal_prompts:
            phrase = template_prompt.format(normal_prompt.format(category))
            normal_phrases += [phrase]

        # abnormal prompts
        for abnormal_prompt in state_level_abnormal_prompts:
            phrase = template_prompt.format(abnormal_prompt.format(category))
            abnormal_phrases += [phrase]

    normal_phrases = tokenize(normal_phrases).to(device)
    abnormal_phrases = tokenize(abnormal_phrases).to(device)
    
    normal_text_features = model(normal_phrases)
    abnormal_text_features = model(abnormal_phrases)
    
    # visualize the group of features
    prompt_feature_visualize(normal_text_features, abnormal_text_features)
    
    text_features = []
    normal_text_features /= normal_text_features.norm(dim=-1, keepdim=True)
    normal_text_features = normal_text_features.mean(dim=0)
    normal_text_features /= normal_text_features.norm()
    text_features.append(normal_text_features)
    
    abnormal_text_features /= abnormal_text_features.norm(dim=-1, keepdim=True)
    abnormal_text_features = abnormal_text_features.mean(dim=0)
    abnormal_text_features /= abnormal_text_features.norm()
    text_features.append(abnormal_text_features)
    
    text_features = torch.stack(text_features, dim=1).to(device).t()
    # apply softmax to text_features category
    # text_features = text_features.softmax(dim=0)
    return [text_features]


def encode_text_with_prompt_ensemble_anomaly_category(model, category, device):
    
    template_level_prompts = [
        'a cropped photo of the {}',
        'a cropped photo of a {}',
        'a close-up photo of a {}',
        'a close-up photo of the {}',
        'a bright photo of a {}',
        'a bright photo of the {}',
        'a dark photo of the {}',
        'a dark photo of a {}',
        'a jpeg corrupted photo of a {}',
        'a jpeg corrupted photo of the {}',
        'a blurry photo of the {}',
        'a blurry photo of a {}',
        'a photo of a {}',
        'a photo of the {}',
        'a photo of a small {}',
        'a photo of the small {}',
        'a photo of a large {}',
        'a photo of the large {}',
        'a photo of the {} for visual inspection',
        'a photo of a {} for visual inspection',
        'a photo of the {} for anomaly detection',
        'a photo of a {} for anomaly detection'
    ]
    
    state_level_normal_prompts = [
        '{}',
        'flawless {}',
        'perfect {}',
        'unblemished {}',
        '{} without flaw',
        '{} without defect',
        '{} without damage'
        #'three flawless, perfect and unblemished {} with different colors without any defect, flaw or damage in a bowl',
        #    'flawless, perfect and unblemished {} without any defect, flaw or damage'
    ]

    state_level_abnormal_prompts = [
        #   'damaged {} with flaw or defect or damage',
            'damaged {}',
            '{} with flaw',
            '{} with defect',
            '{} with damage',
        ##   '{} with missing parts'
        ##   '{} with print',  # added
        ##    '{} with hole',  # added
        ##   '{} with crack', # added
        ##   '{} with scratch', # added
        ##    '{} with discoloration',
        ##    '{} with stains',
        ##    '{} with missing parts',
        ##    '{} with broken parts',
        ##    '{} with bumpy surfaces'
    ]

    text_features = []
    normal_text_features = []
    abnormal_text_features = []
    tot_nomral_text_features = []
    tot_abnormal_text_features = []
    # some categories can be renamed to generate better embedding
    #if category == 'grid':
    #    category  = 'chain-link fence'
    #if category == 'toothbrush':
    #    category = 'brush' #'brush' #
    for template_prompt in template_level_prompts:
        template_text_features = []
        # normal prompts
        template_normal_phrases = []
        for normal_prompt in state_level_normal_prompts:
            phrase = template_prompt.format(normal_prompt.format(category))
            template_normal_phrases += [phrase]

        # abnormal prompts
        template_anormal_phrases = []
        for abnormal_prompt in state_level_abnormal_prompts:
            phrase = template_prompt.format(abnormal_prompt.format(category))
            template_anormal_phrases += [phrase]

        template_normal_phrases = tokenize(template_normal_phrases).to(device)
        template_anormal_phrases = tokenize(template_anormal_phrases).to(device)
    
        template_normal_text_features = model(template_normal_phrases)
        template_abnormal_text_features = model(template_anormal_phrases)
        
        tot_nomral_text_features.append(template_normal_text_features)
        tot_abnormal_text_features.append(template_abnormal_text_features)
        
        template_normal_text_features /= template_normal_text_features.norm(dim=-1, keepdim=True)
        template_normal_text_features = template_normal_text_features.mean(dim=0)
        template_normal_text_features /= template_normal_text_features.norm()
        template_text_features.append(template_normal_text_features)
        normal_text_features.append(template_normal_text_features)
        
        template_abnormal_text_features /= template_abnormal_text_features.norm(dim=-1, keepdim=True)
        template_abnormal_text_features = template_abnormal_text_features.mean(dim=0)
        template_abnormal_text_features /= template_abnormal_text_features.norm()
        template_text_features.append(template_abnormal_text_features)
        abnormal_text_features.append(template_abnormal_text_features)
        
        
        template_text_features = torch.stack(template_text_features, dim=1).to(device).t()
        
        text_features.append(template_text_features)    

    tot_nomral_text_features = torch.cat(tot_nomral_text_features, dim=0).to(device)
    tot_abnormal_text_features = torch.cat(tot_abnormal_text_features, dim=0).to(device)
    # visualize the group of features
    # prompt_feature_visualize(normal_text_features, abnormal_text_features, tot_nomral_text_features, tot_abnormal_text_features)
    centroid_normal_text_features, centroid_abnormal_text_features = centroid_feature_engineer(normal_text_features, abnormal_text_features)
    prompt_feature_visualize_4_groups(normal_text_features, abnormal_text_features, tot_nomral_text_features, tot_abnormal_text_features, centroid_normal_text_features, centroid_abnormal_text_features)
    
    return text_features

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
def prompt_feature_visualize(group1_features, group2_features, tot_group1_features=None, tot_group2_features=None):
    group1_features = torch.stack(group1_features, dim=0)
    group2_features = torch.stack(group2_features, dim=0)
    group1_features_np = group1_features.detach().cpu().numpy()
    group2_features_np = group2_features.detach().cpu().numpy()
    
    all_features_np = np.concatenate((group1_features_np, group2_features_np), axis=0)


    # Labels to differentiate the two groups (0 for group 1, 1 for group 2)
    labels = np.array([0] * group1_features_np.shape[0] + [1] * group2_features_np.shape[0])

    # Perform t-SNE to get 2D embeddings
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(all_features_np)

    # Separate the 2D embeddings back into two groups
    group1_embeddings = embeddings_2d[labels == 0]
    group2_embeddings = embeddings_2d[labels == 1]
    
    # Create a scatter plot with different colors for each group
    plt.scatter(group1_embeddings[:, 0], group1_embeddings[:, 1], label='Normal', c='b')
    plt.scatter(group2_embeddings[:, 0], group2_embeddings[:, 1], label='Anomaly', c='r')
    
    
    for i in range(group1_embeddings.shape[0]):
        plt.arrow(group1_embeddings[i, 0], group1_embeddings[i, 1],
                group2_embeddings[i, 0] - group1_embeddings[i, 0],
                group2_embeddings[i, 1] - group1_embeddings[i, 1],
                color='gray', alpha=0.5, width=0.005)
    
    plt.title('2D t-SNE Visualization of Prompt Feature Embeddings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig("prompt_features_tSNE.png")
    
    
def prompt_feature_visualize_4_groups(
        group1_features, 
        group2_features, 
        tot_group1_features=None, 
        tot_group2_features=None, 
        centroid_normal_text_features=None, 
        centroid_abnormal_text_features=None
    ):
    
    group1_features = torch.stack(group1_features, dim=0)
    group2_features = torch.stack(group2_features, dim=0)
    group1_mean = torch.mean(group1_features, dim=0).unsqueeze(0)
    group2_mean = torch.mean(group2_features, dim=0).unsqueeze(0)
    group1_features_np = group1_features.detach().cpu().numpy()
    group2_features_np = group2_features.detach().cpu().numpy()
    group1_mean_np = group1_mean.detach().cpu().numpy()
    group2_mean_np = group2_mean.detach().cpu().numpy()
    tot_group1_features = tot_group1_features.detach().cpu().numpy()
    tot_group2_features = tot_group2_features.detach().cpu().numpy()
    
    all_features_np = np.concatenate((group1_features_np, group2_features_np, 
                                      tot_group1_features, tot_group2_features, 
                                      group1_mean_np, group2_mean_np,
                                      centroid_normal_text_features, centroid_abnormal_text_features), 
                                     axis=0)

    # Labels to differentiate the two groups (0 for group 1, 1 for group 2)
    labels = np.array([0] * group1_features_np.shape[0] + [1] * group2_features_np.shape[0] + 
                      [2] * tot_group1_features.shape[0] + [3] * tot_group2_features.shape[0] + 
                      [4] * group1_mean_np.shape[0] + [5] * group2_mean_np.shape[0] + 
                      [6] * centroid_normal_text_features.shape[0] + [7] * centroid_abnormal_text_features.shape[0])

    # Perform t-SNE to get 2D embeddings
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(all_features_np)

    # Separate the 2D embeddings back into two groups
    group1_embeddings = embeddings_2d[labels == 0]
    group2_embeddings = embeddings_2d[labels == 1]
    tot_group1_embeddings = embeddings_2d[labels == 2]
    tot_group2_embeddings = embeddings_2d[labels == 3]
    group1_mean_embeddings = embeddings_2d[labels == 4]
    group2_mean_embeddings = embeddings_2d[labels == 5]
    group1_centroid_embeddings = embeddings_2d[labels == 6]
    group2_centroid_embeddings = embeddings_2d[labels == 7]
    
    # Create a scatter plot with different colors for each group
    plt.scatter(group1_embeddings[:, 0], group1_embeddings[:, 1], label='Normal', c='b', marker='o')
    plt.scatter(group2_embeddings[:, 0], group2_embeddings[:, 1], label='Anomaly', c='r', marker='o')
    plt.scatter(tot_group1_embeddings[:, 0], tot_group1_embeddings[:, 1], label='Normal Cluster Centroid', c='b',  marker='x')
    plt.scatter(tot_group2_embeddings[:, 0], tot_group2_embeddings[:, 1], label='Anomaly Cluster Centroid', c='r',  marker='x')
    plt.scatter(group1_mean_embeddings[:, 0], group1_mean_embeddings[:, 1], label='Normal Centroid', c='g',  marker='x', linewidth=2, s=100)
    plt.scatter(group2_mean_embeddings[:, 0], group2_mean_embeddings[:, 1], label='Anomaly Centroid', c='k',  marker='x', linewidth=2, s=100)
    plt.scatter(group1_centroid_embeddings[:, 0], group1_centroid_embeddings[:, 1], label='Anomaly Centroid', c='g',  marker='D', linewidth=2, s=100)
    plt.scatter(group2_centroid_embeddings[:, 0], group2_centroid_embeddings[:, 1], label='Anomaly Centroid', c='k',  marker='D', linewidth=2, s=100)
    
    
    # for i in range(group1_embeddings.shape[0]):
    #     plt.arrow(group1_embeddings[i, 0], group1_embeddings[i, 1],
    #             group2_embeddings[i, 0] - group1_embeddings[i, 0],
    #             group2_embeddings[i, 1] - group1_embeddings[i, 1],
    #             color='gray', alpha=0.5, width=0.005)
    
    plt.title('2D t-SNE Visualization of Prompt Feature Embeddings total')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig("prompt_features_tSNE_tot_mean_centroid.png")
    
    
def centroid_feature_engineer(normal_text_features, abnormal_text_features):
    centroids_A = (torch.stack(normal_text_features, dim=0)).detach().cpu().numpy()
    centroids_B = (torch.stack(abnormal_text_features, dim=0)).detach().cpu().numpy()

    # Calculate pairwise distances
    distances = np.linalg.norm(centroids_A[:, np.newaxis] - centroids_B, axis=2)
    
    # Calculate weights based on distances
    weights_A = 1 / distances.sum(axis=1)
    weights_B = 1 / distances.sum(axis=0)
    
    # Calculate weighted centroids
    weighted_centroid_A = np.sum(centroids_A * weights_A[:, np.newaxis], axis=0)
    weighted_centroid_B = np.sum(centroids_B * weights_B[:, np.newaxis], axis=0)
    
    return weighted_centroid_A.reshape(1, -1), weighted_centroid_B.reshape(1, -1)
    