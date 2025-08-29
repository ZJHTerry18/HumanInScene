<div align="center">
  <h1> HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding </h1>
</div>

*HIS-GPT* is a large multi-modal foundation model for **human-in-scene (HIS)** understanding, a new task that we raise for understanding human behaviors in 3D scenes. To evaluate this new task, we also release *HIS-Bench*, the first multi-modal benchmark for comprehensively evaluating model's abilities on human-in-scene understanding. [<a href="https://arxiv.org/abs/2503.12955">Paper</a>]

![overview.png](assets/overview.png)

**TODO**:

- [x] Upload the training & evaluation code.
- [x] Release the annotations of HIS-Bench and HIS-GPT training data.
- [ ] Release the pretrained weights of HIS-GPT.

## HIS-Bench


## HIS-GPT

### Quick Start
<details>
  <summary><b>Environmental Setup</b></summary>
  
```
conda create -n hisgpt python=3.10
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
</details>
<details>
  <summary><b>Data Preparation</b></summary>
  
  Download the HIS-GPT training data from <a href="https://uob-my.sharepoint.com/:f:/g/personal/ur25890_bristol_ac_uk/En2epSoEPBRLpy7qCL2PxfIBrocUlIzniZdMarmjjBekAA?e=qYil1E">here</a>.
  
  Put all the data under the `./annotations` directory. Unzip the `.zip` files in the subdirectories. You will get the directory contains the following contents:
```
annotations
├── scannet_mask3d_uni3d_feats.pt      # 3D scene representations for ScanNet scenes (used by HUMANISE and SceneVerse)
├── scannet_mask3d_train_attributes.pt # 3D scene attributes for ScanNet scenes (used by HUMANISE and SceneVerse)
├── trumans_mask3d_uni3d_feats.pt      # 3D scene representations for TRUMANS scenes
├── trumans_mask3d_train_attributes.pt # 3D scene attributes for TRUMANS scenes
├── m3gpt_t2m_motion_embeds.pt         # embedding vectors for human motions
├── humanise/trumans                   # annotations for human-in-scene data
    ├── qas_pt_v1    # HUMANISE captions for pre-training
    ├── qas_train_v1 # HUMANISE QA data for instruction tuning
    ├── motion_tokens # tokens for 3D human motions
    └── motion_trajs  # trajectory for 3D human motions
├── sceneverse                         # annotations for SceneVerse (scene-only) data
└── motionx                            # annotations for HumanML3D (motion-only) data
```
For 3D scene and 3D human motion data, we pre-extracted them into latent embeddings using the relevant encoders (to save storage). That is, the features and attributes in our provided annotations are directly fed into the projection layers and the large language model when you run the training codes.

Note: If you want to extract 3D scene features (`..._uni3d_feats.pt` and `..._train_attributes.pt`) from the raw data, you could refer to <a href="https://github.com/ZzZZCHS/Chat-Scene/tree/dev/preprocess">this guidance</a>.
</details>

<details>
  <summary><b>Model Preparation</b></summary>
  
  Download <a href="https://huggingface.co/lmsys/vicuna-7b-v1.5">vicuna-7b-v1.5</a>, which is the model we will use as the pre-trained LLM.

</details>

### Training
- Configurations before training
  - Set the `llama_model_path` in `scripts/human_scene_pt.sh` and `scripts/human_scene_it.sh` to your own vicuna-7b-v1.5 path.
  - Set `output_dir` to your own output directory.

*Step 1*: Multi-modal pre-training:
```
bash scripts/human_scene_pt.sh
```

*Step 2*: Human-in-scene instruction tuning:
```
bash scripts/human_scene_it.sh
```

### Evaluation
To evaluate the model on HIS-Bench, please first download and prepare the HIS-Bench data according to the following steps:

Then, run the inference to get the model's answers on HIS-Bench questions:

Finally, use the GPT-based evaluation code to get the performance score:

## Citation
If you find our paper useful, please consider citing:
```{bibtex}
@misc{zhao2025hisgpt3dhumaninscenemultimodal,
      title={HIS-GPT: Towards 3D Human-In-Scene Multimodal Understanding}, 
      author={Jiahe Zhao and Ruibing Hou and Zejie Tian and Hong Chang and Shiguang Shan},
      year={2025},
      eprint={2503.12955},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12955}, 
}
```

## Acknowledgements
This code implementation is based on [Chat-Scene](https://github.com/ZzZZCHS/Chat-Scene). Thanks to their awesome work!
