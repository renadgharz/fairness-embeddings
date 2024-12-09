import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder

class SCINDataset(Dataset):

    def __init__(self, root_dir, labels_csv, cases_csv, transform=None, protected_attr='combined_race'):
        
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(os.path.join(root_dir, labels_csv))
        self.cases_df = pd.read_csv(os.path.join(root_dir, cases_csv))
        self.transform = transform
        self.protected_attr_name = protected_attr

        self.labels_df['case_id'] = self.labels_df['case_id'].astype(str).str.strip()
        self.cases_df['case_id'] = self.cases_df['case_id'].astype(str).str.strip()
        self.data = pd.merge(self.labels_df, self.cases_df, on="case_id")
        self.images_dir = os.path.join(root_dir, "images")

        self.data['image_filename'] = self.data['image_1_path'].apply(lambda x: os.path.basename(x))
        self.data = self.data[
            self.data['image_filename'].notna() &
            self.data['image_filename'].apply(lambda x: os.path.exists(os.path.join(self.images_dir, x)))
        ]

        self.label_encoder = LabelEncoder()
        self.data['encoded_label'] = self.label_encoder.fit_transform(self.data['weighted_skin_condition_label'])

        self.protected_label_encoder = LabelEncoder()
        self.data['encoded_protected_attr'] = self.protected_label_encoder.fit_transform(self.data[self.protected_attr_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_dir, row['image_filename'])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['encoded_label'], dtype=torch.long)
        protected_attr = torch.tensor(row['encoded_protected_attr'], dtype=torch.long)

        return image, label, protected_attr

class HAM10000Dataset(Dataset):
    
    def __init__(self, root_dir, metadata_csv, transform=None, protected_attr='sex'):

        self.root_dir = root_dir
        self.metadata_path = os.path.join(root_dir, metadata_csv)
        self.transform = transform
        self.protected_attr_name = protected_attr
        
        self.metadata = pd.read_csv(self.metadata_path)
        
        self.metadata['image_path'] = self.metadata['image_id'].apply(self._get_image_path)
        self.metadata = self.metadata[self.metadata['image_path'].notna()]

        self.label_encoder = LabelEncoder()
        self.metadata['encoded_label'] = self.label_encoder.fit_transform(self.metadata['dx'])

        self.protected_label_encoder = LabelEncoder()
        self.metadata['encoded_protected_attr'] = self.protected_label_encoder.fit_transform(
            self.metadata[self.protected_attr_name].fillna('Unknown') 
        )

    def _get_image_path(self, image_id):

        for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
            image_path = os.path.join(self.root_dir, folder, f"{image_id}.jpg")
            if os.path.exists(image_path):
                return image_path
        return None 

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        
        row = self.metadata.iloc[idx]
        image_path = row['image_path']
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row['encoded_label'], dtype=torch.long)
        protected_attr = torch.tensor(row['encoded_protected_attr'], dtype=torch.long)

        return image, label, protected_attr

