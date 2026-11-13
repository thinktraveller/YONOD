# Model weights

The distributable model binaries are hosted on Hugging Face rather than in
this Git repository:

<https://huggingface.co/thinktraveller/YONOD-weights>

To restore the `WEIGHTS/` layout in a YONOD checkout, authenticate with the
Hugging Face CLI if required and run:

Linux Bash (with the `yonod` environment active), from the repository root:

```bash
conda activate yonod
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

Windows Conda PowerShell, from the repository root:

```powershell
conda activate yonod
hf download thinktraveller/YONOD-weights --local-dir WEIGHTS
```

The download command is the same on both systems. Install the compatible Hugging Face CLI and initialize Conda as described in the [main installation guide](../README.md).

The downloaded binary weight files are ignored by Git. Review the licenses and
terms of their upstream models before reuse.
