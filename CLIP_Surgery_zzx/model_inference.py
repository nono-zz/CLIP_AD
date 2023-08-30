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

    
    return text_features
