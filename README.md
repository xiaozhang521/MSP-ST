
# MSP-ST (Multi step pre-training for speech translation)

This is the source code of paper [Improving End-to-end Speech Translation by Leveraging Auxiliary Speech and Text Data](https://arxiv.org/pdf/2212.01778.pdf).

The code is forked from `Fairseq-v0.12.2`, please refer to [`Fairseq`](https://github.com/facebookresearch/fairseq/tree/v0.12.2#requirements-and-installation) for more Installation details.

## Useage

There are `three steps` to pretrain the speech translation model. The training scripts and configurations on `MuST-C` data-set are listed following:
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

- Download the pre-trained [`MBART`](https://github.com/facebookresearch/fairseq/tree/v0.12.2/examples/mbart) model. Use the `mt-pretrain/preprocess.sh` to process the parallel data. Note that the data should be cut into pieces by `Sentencepiece` toolkit.


- Set the path of `MBART` model and processed paralled data in script `mt-pretrain/pretrain.sh`, then run it.


- To save the model parameters, you can use the `mt-pretrain/load_embedding.py` to fetch necessary word embeddings.


### Step2 ASR pre-training

- To process the ASR data, please follow [here](https://github.com/facebookresearch/fairseq/tree/v0.12.2/examples/wav2vec#prepare-training-data-manifest). 


- Change all the required paths in the `asr-pretrain/config.yaml`, then run the `asr-pretrain/asr_pretrain.sh`.


### Step3 ST fine-tuning

- To process the `MuST-C` ST data, please follow [here](https://github.com/facebookresearch/fairseq/blob/v0.12.2/examples/speech_to_text/docs/mustc_example.md#data-preparation).


- Set model paths of `step 1` and `step 2` in `st-finetune/conf/finetune.yaml` and set the data path in `st-finetune/st_finetune.sh`. 


### Decode
Run the `st-finetune/decode.sh` for inference.

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
