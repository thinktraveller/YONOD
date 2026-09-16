# Model weights

The distributable model binaries are hosted on Hugging Face rather than in
this Git repository:

<https://huggingface.co/thinktraveller/YONOD-weights>

To restore the `WEIGHTS/` layout in a YONOD checkout, authenticate with the
Hugging Face CLI if required and run:

```bash
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

The downloaded binary weight files are ignored by Git. Review the licenses and
terms of their upstream models before reuse.
