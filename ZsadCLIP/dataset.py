import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os


class VisaDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]
		self.size_list = []

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		img = Image.open(os.path.join(self.root, img_path))
		if img.size not in self.size_list:
			self.size_list.append(img.size)
		if anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
		else:
			img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class MVTecDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		name = self.root.split('/')[-1]
		meta_info = meta_info[mode]

		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = list(meta_info.keys())
		for cls_name in self.cls_names:
			if mode == 'train':
				data_tmp = meta_info[cls_name]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			else:
				self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			random_defect = random.choice(defect)
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
			else:
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask

	def __getitem__(self, index):
		data = self.data_all[index]
		img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
															  data['specie_name'], data['anomaly']
		random_number = random.random()
		if random_number < self.aug_rate:
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, img_path))
			if anomaly == 0:
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
			else:
				img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
		# transforms
		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
				'img_path': os.path.join(self.root, img_path)}


class MvTecLocoDataset(data.Dataset):
	def __init__(self, root, transform, target_transform, aug_rate=-1, mode='test', k_shot=0, save_dir=None, obj_name=None):
		self.root = root
		self.transform = transform
		self.target_transform = target_transform
		self.aug_rate = aug_rate

		self.data_all = []
		# meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
		# name = self.root.split('/')[-1]
		# meta_info = meta_info[mode]
		self.tot_class_names = os.listdir(root)
		self.tot_class_names = [item for item in self.tot_class_names if os.path.isdir(os.path.join(root, item))]


		if mode == 'train':
			self.cls_names = [obj_name]
			save_dir = os.path.join(save_dir, 'k_shot.txt')
		else:
			self.cls_names = self.tot_class_names
		for cls_name in self.cls_names:
			if mode == 'train':
				cls_train_dir = os.path.join(root, cls_name, 'train/good')
				data_tmp = [os.path.join(cls_train_dir, img_name) for img_name in os.listdir(cls_train_dir)]
				indices = torch.randint(0, len(data_tmp), (k_shot,))
				for i in range(len(indices)):
					self.data_all.append(data_tmp[indices[i]])
					with open(save_dir, "a") as f:
						f.write(data_tmp[indices[i]]['img_path'] + '\n')
			# test mode: load all test data images
			else:
				for cls_test_category in os.listdir(os.path.join(root, cls_name, 'test')):
					self.data_all.extend([os.path.join(root, cls_name, 'test', cls_test_category, img_name) 
                           for img_name in os.listdir(os.path.join(root, cls_name, 'test', cls_test_category))])       
				# self.data_all.extend(meta_info[cls_name])
		self.length = len(self.data_all)

	def __len__(self):
		return self.length

	def get_cls_names(self):
		return self.cls_names

	def combine_img(self, cls_name):
		img_paths = os.path.join(self.root, cls_name, 'test')
		img_ls = []
		mask_ls = []
		for i in range(4):
			defect = os.listdir(img_paths)
			random_defect = random.choice(defect)
			files = os.listdir(os.path.join(img_paths, random_defect))
			random_file = random.choice(files)
			img_path = os.path.join(img_paths, random_defect, random_file)
			mask_path = os.path.join(self.root, cls_name, 'ground_truth', random_defect, random_file[:3] + '_mask.png')
			img = Image.open(img_path)
			img_ls.append(img)
			if random_defect == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
			else:
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
			mask_ls.append(img_mask)
		# image
		image_width, image_height = img_ls[0].size
		result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
		for i, img in enumerate(img_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_image.paste(img, (x, y))

		# mask
		result_mask = Image.new("L", (2 * image_width, 2 * image_height))
		for i, img in enumerate(mask_ls):
			row = i // 2
			col = i % 2
			x = col * image_width
			y = row * image_height
			result_mask.paste(img, (x, y))

		return result_image, result_mask

	def load_data_info(self, img_path):
		anomaly_type = img_path.split('/')[-2]
		cls_name = img_path.split('/')[-4]

		if anomaly_type == 'good':
			mask_path = None
		else:
			path_components = img_path.split('/')
			path_components[-3] = 'ground_truth'
			path_components[-1] = path_components[-1].replace('.png', '')
			# path_components
			mask_dir_path = "/".join(path_components)
			mask_files = os.listdir(mask_dir_path)
			if len(mask_files) == 1:
				mask_path = os.path.join(mask_dir_path, mask_files[0])
			else:
				print('two gt_masks are found!!! class name is {}'.format(cls_name))
				mask_path = os.path.join(mask_dir_path, mask_files[0])		# leave it here for future changes
		return mask_path, cls_name, anomaly_type

	def __getitem__(self, index):
		img_path = self.data_all[index]
		# img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
		# 													  data['specie_name'], data['anomaly']
		mask_path, cls_name, anomaly_type = self.load_data_info(img_path)
  
		random_number = random.random()
		if random_number < self.aug_rate:
			img, img_mask = self.combine_img(cls_name)
		else:
			img = Image.open(os.path.join(self.root, img_path))
			if anomaly_type == 'good':
				img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
				anomaly = 0
			else:
				img_mask = np.array(Image.open(mask_path).convert('L')) > 0
				img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
				anomaly = 1
		# transforms
		ori_size = img.size
		img = self.transform(img) if self.transform is not None else img
		assert img.shape[-1] == 240 
		img_mask = self.target_transform(
			img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask
		return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly_type': anomaly_type,
				'img_path': os.path.join(self.root, img_path), 'anomaly': anomaly, 'img_ori_size': ori_size}


import random
import string
class RandomWordPrompt(data.Dataset):
    def __init__(self, num_sentences, model, tokenizer, device, args, length_min=5, length_max=10):
        self.num_sentences = num_sentences
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.length_min = length_min
        self.length_max = length_max
        self.args = args
        self.normal_states = ['a', 'an', 'a normal', 'a good', 'a flawless', 'a perfect', 'a unblemished']
        self.anomaly_states = ['a damaged', 'a borken', 'a defective', 'an anomalous', 'an imperfect', 'a blemished', 'an abnormal']
        
    def generate_random_word(self, length):
        characters = string.ascii_lowercase + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def __len__(self):
        return self.num_sentences
    
    def generate_prompt(self):
        # normal_prompt_template = '{} a {} photo {} of {} a {}'
        if self.args.multiple_states:
            normal_state = random.choice(self.normal_states)
            abnormal_state = random.choice(self.anomaly_states)
        else:
            normal_state = 'a'
            abnormal_state = 'a damaged'
            
        normal_prompt_template = '{} a {} photo {} of {} {} {}'
        anomaly_prompt_template = '{} a {} photo {} of {} {} {}'
    
        prompt_list = []
        
        word1 = self.generate_random_word(random.randint(5, 10))
        word2 = self.generate_random_word(random.randint(5, 10))
        word3 = self.generate_random_word(random.randint(5, 10))
        word4 = self.generate_random_word(random.randint(5, 10))
        word5 = self.generate_random_word(random.randint(5, 10))
        
        word6 = self.generate_random_word(random.randint(5, 10))
        word7 = self.generate_random_word(random.randint(5, 10))
        word8 = self.generate_random_word(random.randint(5, 10))
        word9 = self.generate_random_word(random.randint(5, 10))
        word10 = self.generate_random_word(random.randint(5, 10))
	
        normal_prompt = normal_prompt_template.format(word1, word2, word3, word4, normal_state, word5)
        anomaly_prompt = anomaly_prompt_template.format(word6, word7, word8, word9, abnormal_state, word10)
        
        prompt_list.append(normal_prompt)
        prompt_list.append(anomaly_prompt)
        
        prompt_tokenized = self.tokenizer(prompt_list).to(self.device)
        prompt_embeddings = self.model.encode_text(prompt_tokenized)
        
        prompt_label_list = [0,1]
        prompt_labels = torch.tensor(prompt_label_list).to(self.device)
        prompt_labels = prompt_labels.to(prompt_embeddings.dtype)
        return prompt_embeddings, prompt_labels
    
    def __getitem__(self, idx):
        prompt_embeddings, prompt_labels = self.generate_prompt()
        return prompt_embeddings, prompt_labels