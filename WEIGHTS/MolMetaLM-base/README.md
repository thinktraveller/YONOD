---
pipeline_tag: text-generation
tags:
- Molecule Language Model
- Physicochemical Knowledge
---

refer to https://github.com/CSUBioGroup/MolMetaLM for more details. 

# Usage

## Prepare tokenizer and model
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('wudejian789/MolMetaLM-base')
model = AutoModel.from_pretrained('wudejian789/MolMetaLM-base')
```

## Obtain molecular representations from SMILES
```python
smi = "COc1cc2c(cc1OC)CC([NH3+])C2"
tokenized_smi = tokenizer(" ".join(list(smi)), return_token_type_ids=False, 
                          return_tensors='pt', max_length=512, padding='longest', truncation=True)
emb_smi = model(**tokenized_smi).last_hidden_state
print(emb_smi.shape) # batch size, seq length, embedding size
```
