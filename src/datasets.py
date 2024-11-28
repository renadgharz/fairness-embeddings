import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder

class SCINDataset(Dataset):
    """
    A class to load the SCIN dataset, including images, labels, and protected attributes.
    """

    def __init__(self, root_dir, labels_csv, cases_csv, transform=None, protected_attr='combined_race'):
        
        """
        Args:
            root_dir (str): Path to the root directory of the dataset.
            labels_csv (str): Name of the CSV file containing the labels.
            cases_csv (str): Name of the CSV file containing the cases.
            transform (callable, optional): A function/transform to apply to the images.
            protected_attr (str): Column name to be used as the protected attribute.
        """
        
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
