
# MSP-ST (Multi step pre-training for speech translation)

This is the source code of paper [Improving End-to-end Speech Translation by Leveraging Auxiliary Speech and Text Data](https://arxiv.org/pdf/2212.01778.pdf).

The code is fork from Fairseq-v0.12.2, please refer to [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.12.2#requirements-and-installation) for more install details.

## Useage

There are three steps to pretrain the speech translation model. The training scripts and configurations on MuST-C data-set are listed following:
```
egs
|---mt-pretrain
|       |---preprocess.sh
|       |---pretrain.sh
|       |---load_embedding.py
|---asr-pretrain
|       |---asr_pretrain.sh
|       |---config.yaml
|---st-finetune
|       |---st_finetune.sh
|       |---decode.sh
|       |---conf
```
### Step1 MT pre-training

Download the pre-trained [MBART](https://github.com/facebookresearch/fairseq/tree/v0.12.2/examples/mbart) model. Use the _preprocess.sh_ to process the parallel data. Note the data should be cut into piece by Sentencepiece toolkit.

Set the path to download model and data path in script _pretrain.sh_ then run it.

To save the model parameters, you can use the _load_embedding.py_ to fetch necessary word embeddings.

### Step2 ASR pre-training

To process the ASR data, please follow [here](https://github.com/facebookresearch/fairseq/tree/v0.12.2/examples/wav2vec#prepare-training-data-manifest). 

Change all the required paths in the _config.yaml_ then run the _asr_pretrain.sh_.

### Step3 ST fine-tuning

To process the MuST-C ST data, please follow [here](https://github.com/facebookresearch/fairseq/blob/v0.12.2/examples/speech_to_text/docs/mustc_example.md#data-preparation).

Set model paths of step 2 in _conf/finetune.yaml_ and set the data path in _st_finetune.sh_. 

### Decode
Run the _decode.sh_ to inference.

## Citation

If you want to cite the paper, please cite as:

``` bibtex
@article{zhang2022improving,
  title={Improving End-to-end Speech Translation by Leveraging Auxiliary Speech and Text Data},
  author={Zhang, Yuhao and Xu, Chen and Hu, Bojie and Zhang, Chunliang and Xiao, Tong and Zhu, Jingbo},
  journal={arXiv preprint arXiv:2212.01778},
  year={2022}
}
```
