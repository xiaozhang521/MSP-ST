import torch
from fairseq.models.bart import BARTModel
import numpy as np
mbart = BARTModel.from_pretrained('path to mt model', checkpoint_file='last5.ensemble.pt',data_name_or_path="path to training data",sentencepiece_model="path to sentence model")
mbart.eval()
embedding=mbart.model.decoder.embed_tokens.weight.data
dicts=open("the path to mt dict","r").read().strip().split("\n")
dicts=[i.split(" ")[0] for i in dicts]
small_dicts=open("the path to st dict","r").read().strip().split("\n")
small_dicts=[i.split(" ")[0] for i in small_dicts]
output=open("pretrain_embeddings_enfr","w")
dicts=["<s>","<pad>","</s>","<unk>"]+dicts
small_dicts=["<s>","<pad>","</s>","<unk>"]+small_dicts
dicts=dicts+["[ar_AR]","[cs_CZ]","[de_DE]","[en_XX]","[es_XX]","[et_EE]","[fi_FI]","[fr_XX]","[gu_IN]","[hi_IN]","[it_IT]","[ja_XX]","[kk_KZ]","[ko_KR]","[lt_LT]","[lv_LV]","[my_MM]","[ne_NP]","[nl_XX]","[ro_RO]","[ru_RU]","[si_LK]","[tr_TR]","[vi_VN]","[zh_CN]"]
small_dicts=small_dicts+["[ar_AR]","[cs_CZ]","[de_DE]","[en_XX]","[es_XX]","[et_EE]","[fi_FI]","[fr_XX]","[gu_IN]","[hi_IN]","[it_IT]","[ja_XX]","[kk_KZ]","[ko_KR]","[lt_LT]","[lv_LV]","[my_MM]","[ne_NP]","[nl_XX]","[ro_RO]","[ru_RU]","[si_LK]","[tr_TR]","[vi_VN]","[zh_CN]"]
output.write(str(len(small_dicts))+" 1024\n")
for index in range(len(dicts)):
    if dicts[index] in small_dicts:
        output.write(dicts[index]+" ")
        feature=embedding[index:index+1,].numpy()
        np.savetxt(output,feature)
 


