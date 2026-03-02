#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rfdetr import RFDETRBase

# Modèle pré-entraîné COCO
model = RFDETRBase()

model.train(
    dataset_dir="../../../../Dataset2D/COCO_Dataset",   # dossier racine contenant train/ et val/
    epochs=60,
    batch_size=4,
    lr=1e-4,
    output_dir="../../../../results/rfdetr/run1"
)


# In[ ]:




