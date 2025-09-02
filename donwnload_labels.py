# pip install segments-ai
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset

# load key from user environment variable
import os
api_key = os.environ['SEGMENTS']

# Initialize a SegmentsDataset from the release file
client = SegmentsClient(api_key)
release = client.get_release('TimoWalraven/pools', 'v0.1') # Alternatively: release = 'flowers-v1.0.json'
dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

# Export to COCO panoptic format
export_dataset(dataset, export_format='coco-panoptic')