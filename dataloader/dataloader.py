import torch, os, json, re, numpy as np, torch.nn as nn, random
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, BertModel
class DTD2(torch.utils.data.Dataset):
    def __init__(self, root_dir, descriptions, splits, split=None):
        self.root_dir = root_dir
        self.split = split
        self.splits = json.load(open(os.path.join(root_dir, splits))) # json file
        self.descriptions = json.load(open(os.path.join(root_dir, descriptions))) # json file
        self.transform = None
        self.data = TextureDescriptionData()
    def __getitem__(self, index): # return (augmented_image, label)
        if self.split == None:
            return None
        elif self.split == 'train':
            return self.read_image(self.splits['train'][index], True)
        elif self.split == 'test':
            return self.read_image(self.splits['test'][index], False)
        elif self.split == 'val':
            return self.read_image(self.splits['val'][index], False)

    def __len__(self): # returns length of current split.
        if self.split == None:
            return len(self.splits['train']) + len(self.splits['test']) + len(self.splits['val'])
        elif self.split == 'train':
            return len(self.splits['train'])
        elif self.split == 'test':
            return len(self.splits['test'])
        elif self.split == 'val':
            return len(self.splits['val'])
    
    def read_image(self, image_location, is_train): # open and transform image, return image
        # image
        img = Image.open(os.path.join(self.root_dir, image_location)).convert('RGB')
        self.transform = self.transformations(training=is_train)
        # label
        label = torch.zeros(len(self.data.phrases))
        for i in self.data.img_data_dict[image_location]['phrase_ids']:
            if i >= 0:
                label[i] = 1
        return self.transform(img), label            

    def transformations(self, training=True): # standard image transforms from DTD2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if training:
            return transforms.Compose([ transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            return transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

class DTD2gen(torch.utils.data.Dataset):
    def __init__(self, root_dir, descriptions, splits, split=None):
        self.root_dir = root_dir
        self.split = split
        self.splits = json.load(open(os.path.join(root_dir, splits))) # json file
        self.descriptions = json.load(open(os.path.join(root_dir, descriptions))) # json file
        self.transform = None
        self.data = TextureDescriptionData()
    def __getitem__(self, index): # return (augmented_image, label)
        if self.split == None:
            return None
        elif self.split == 'train':
            return self.read_image(self.splits['train'][index], True)
        elif self.split == 'test':
            return self.read_image(self.splits['test'][index], False)
        elif self.split == 'val':
            return self.read_image(self.splits['val'][index], False)

    def __len__(self): # returns length of current split.
        if self.split == None:
            return len(self.splits['train']) + len(self.splits['test']) + len(self.splits['val'])
        elif self.split == 'train':
            return len(self.splits['train'])
        elif self.split == 'test':
            return len(self.splits['test'])
        elif self.split == 'val':
            return len(self.splits['val'])
    
    def read_image(self, image_location, is_train): # open and transform image, return image
        # image
        img = Image.open(os.path.join(self.root_dir, image_location)).convert('RGB')
        self.transform = self.transformations(training=is_train)
        # label
        label = torch.zeros(len(self.data.phrases))
        for i in self.data.img_data_dict[image_location]['phrase_ids']:
            if i >= 0:
                label[i] = 1
        return self.transform(img), label

    def get_descriptions(self, image_name):
        for i in self.descriptions:
            if i['image_name'] == image_name:
                return i['descriptions']

    def transformations(self, training=True): # standard image transforms from DTD2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if training:
            return transforms.Compose([ transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            return transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

class DTD2Triple(torch.utils.data.Dataset):
    def __init__(self, root_dir, descriptions, splits, split=None):
        self.root_dir = root_dir
        self.split = split
        self.splits = json.load(open(os.path.join(root_dir, splits))) # json file
        self.descriptions = json.load(open(os.path.join(root_dir, descriptions))) # json file
        self.transform = None
        self.data = TextureDescriptionData()
        self.max_idx = 0
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
        self.linear = nn.Linear(768, 256) # hardcoded
        self.count = 0
    def __getitem__(self, index): # return (augmented_image, label)
        if self.split == None:
            return None
        elif self.split == 'train':
            return self.read_image(self.splits['train'][index], True, index)
        elif self.split == 'test':
            return self.read_image(self.splits['test'][index], False, index)
        elif self.split == 'val':
            return self.read_image(self.splits['val'][index], False, index)

    def __len__(self): # returns length of current split.
        if self.split == None:
            return len(self.splits['train']) + len(self.splits['test']) + len(self.splits['val'])
        elif self.split == 'train':
            return len(self.splits['train'])
        elif self.split == 'test':
            return len(self.splits['test'])
        elif self.split == 'val':
            return len(self.splits['val'])
    
    def read_image(self, image_location, is_train, index): # open and transform image, return image
        self.transform = self.transformations(training=is_train)
        # anchor farming
        anchor_img = Image.open(os.path.join(self.root_dir, image_location)).convert('RGB')
        descriptions = self.descriptions[index]['descriptions']
        anchor_caption = self.embed_sentences(descriptions)
        label = image_location.split('/')[0]
        # positive farming
        try:
            possible = os.listdir('./dtd/images/' + label)
            rand = random.randint(0, len(possible) - 1)
            pos_img_location = label + '/' + possible[rand]
            while pos_img_location == image_location:
                rand = random.randint(0, len(possible) - 1)
                pos_img_location = label + '/' + possible[rand]
            pos_img = Image.open(os.path.join(self.root_dir, pos_img_location)).convert('RGB')
            pos_caption = self.embed_sentences(self.get_descriptions(pos_img_location))
        except Exception:
            self.count += 1
            pos_img = Image.open(os.path.join(self.root_dir, image_location)).convert('RGB')
            pos_caption = self.embed_sentences(self.get_descriptions(image_location))
            print("Use default as positive " + str(self.count) + " times...")
        # negative farming
        neg_img_location = self.descriptions[random.randint(0, len(self.descriptions)-1)]['image_name']
        while neg_img_location.split('/')[0] == label:
            neg_img_location = self.descriptions[random.randint(0, len(self.descriptions)-1)]['image_name']
        neg_img = Image.open(os.path.join(self.root_dir, neg_img_location)).convert('RGB')
        neg_caption = self.embed_sentences(self.get_descriptions(neg_img_location))
        return self.transform(anchor_img), anchor_caption, self.transform(pos_img), pos_caption, self.transform(neg_img), neg_caption

    def get_descriptions(self, image_name):
        for i in self.descriptions:
            if i['image_name'] == image_name:
                return i['descriptions']

    def embed_sentences(self, descriptions):
        # from: https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
        embeddings = list()
        for sentence in descriptions:
            text = '[CLS] ' + sentence + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(indexed_tokens)
            tokens_tensor, segments_tensors = torch.tensor([indexed_tokens]), torch.tensor([segments_ids])
            outputs = self.model(tokens_tensor, segments_tensors)
            token_vecs = outputs.hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            embeddings.append(sentence_embedding)
        final_embedding = torch.mean(torch.FloatTensor([t.detach().numpy() for t in embeddings]), dim=0)
        with torch.no_grad():
            final_embedding = self.linear(final_embedding)
        return final_embedding
    
    def transformations(self, training=True): # standard image transforms from DTD2
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if training:
            return transforms.Compose([ transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        else:
            return transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])

data_path = '/home/tmisic/texture-descriptions/dtd/images/'
class TextureDescriptionData:
    def __init__(self, phrase_split='train', phrase_freq_thresh=10, phid_format='set'):
        with open(os.path.join(data_path, 'image_splits.json'), 'r') as f:
            self.img_splits = json.load(f)

        self.phrases = list()
        self.phrase_freq = list()
        if phrase_split == 'all':
            phrase_freq_file = 'dataloader/phrase_freq.txt'
        elif phrase_split == 'train':
            phrase_freq_file = 'dataloader/phrase_freq_train.txt'
        else:
            raise NotImplementedError
        with open(phrase_freq_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                phrase, freq = line.split(' : ')
                if int(freq) < phrase_freq_thresh:
                    break
                self.phrases.append(phrase)
                self.phrase_freq.append(int(freq))

        self.phrase_phid_dict = {p: i for i, p in enumerate(self.phrases)}
        self.phid_format = phid_format

        self.img_data_dict = dict()
        with open(os.path.join(data_path, 'image_descriptions.json'), 'r') as f:
            data = json.load(f)
        for img_d in data:
            img_d['phrase_ids'] = self.descpritions_to_phids(img_d['descriptions'])
            self.img_data_dict[img_d['image_name']] = img_d

        self.img_phrase_match_matrices = dict()
        print('TextureDescriptionData ready. \n{ph_num} phrases with frequency above {freq}.\n'
              'Image count: train {train}, val {val}, test {test}'
              .format(ph_num=len(self.phrases), freq=phrase_freq_thresh, train=len(self.img_splits['train']),
                      val=len(self.img_splits['val']), test=len(self.img_splits['test'])))

    def phid_to_phrase(self, phid):
        if phid > len(self.phrases) or phid < 0:
            return '<UNK>'
        return self.phrases[phid]

    def phrase_to_phid(self, phrase):
        return self.phrase_phid_dict.get(phrase, -1)

    @staticmethod
    def description_to_phrases(desc):
        segments = re.split('[,;]', desc)
        phrases = list()
        for seg in segments:
            phrase = seg.strip()
            if len(phrase) > 0:
                phrases.append(phrase)
        return phrases

    def descpritions_to_phids(self, descriptions, phid_format=None):
        if phid_format is None:
            phid_format = self.phid_format

        if phid_format is None:
            return None

        phrases = set()
        if phid_format == 'str':
            for desc in descriptions:
                phrases.update(self.description_to_phrases(desc))
            return phrases

        phids = list()
        for desc in descriptions:
            phrases = self.description_to_phrases(desc)
            phids_desc = [self.phrase_to_phid(ph) for ph in phrases]
            phids.append(phids_desc)

        if phid_format == 'nested_list':
            return phids

        elif phid_format == 'phid_freq':
            phid_freq = dict()
            for phids_desc in phids:
                for phid in phids_desc:
                    phid_freq[phid] = phid_freq.get(phid, 0) + 1
            return phid_freq

        elif phid_format == 'set':
            phid_set = set()
            for phids_desc in phids:
                phid_set.update(phids_desc)
            return phid_set

        else:
            raise NotImplementedError

    def description_to_phids_smart(self, desc):
        phids = set()
        phrases = self.description_to_phrases(desc)
        for ph in phrases:
            if ph in self.phrases:
                phids.add(self.phrase_to_phid(ph))
            else:
                for wd in WordEncoder.tokenize(ph):
                    if wd in self.phrases:
                        phids.add(self.phrase_to_phid(wd))
        return phids

    def get_img_phrase_match_matrices(self, split):
        if split in self.img_phrase_match_matrices:
            return self.img_phrase_match_matrices[split]
        img_num = len(self.img_splits[split])
        phrase_num = len(self.phrases)

        match = np.zeros((img_num, phrase_num), dtype=int)
        for img_i, img_name in enumerate(self.img_splits[split]):
            img_data = self.img_data_dict[img_name]
            if self.phid_format == 'set':
                phid_set = img_data['phrase_ids']
            else:
                phid_set = self.descpritions_to_phids(img_data['descriptions'], phid_format='set')
            for phid in phid_set:
                if phid >= 0:
                    match[img_i, phid] = 1
        self.img_phrase_match_matrices[split] = match
        return match

    def get_gt_phrase_count(self, split):
        gt_phrase_count = np.zeros(len(self.img_splits[split]))
        for img_i, img_name in enumerate(self.img_splits[split]):
            phrases = set()
            img_data = self.img_data_dict[img_name]
            for desc in img_data['descriptions']:
                phrases.update(self.description_to_phrases(desc))
            gt_phrase_count[img_i] = len(phrases)
        return gt_phrase_count