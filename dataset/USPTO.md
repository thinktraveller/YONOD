# USPTO dataset

The cleaned and normalized USPTO corpus is hosted outside this Git repository:

<https://huggingface.co/datasets/thinktraveller/YONOD-datasets>

To restore it under `dataset/USPTO/`, authenticate with the Hugging Face CLI if
required and run:

```bash
hf download thinktraveller/YONOD-datasets --repo-type dataset --include 'USPTO/**' --local-dir dataset
```

The downloaded corpus is ignored by Git. See `USPTO/README.md` after download
for conversion, quality, and provenance details.
