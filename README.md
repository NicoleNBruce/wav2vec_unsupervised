# wav2vec_unsupervised

Wav2vec_unsupervised is a collection of scripts that automate running the Fairseq wav2vec 2.0 Unsupervised Speech Recognition pipeline as described in the official Fairseq project:

https://github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/unsupervised/README.md

These scripts have been tested to work reliably in a Python virtual environment with PyTorch == 2.3.0


## System Requirements

Before running the project, ensure the following requirements are met:

* Linux-based system (recommended)
* NVIDIA GPU with CUDA support
* Python virtual environment (venv)
* Git Installed

### Installing GIT 
if Git is not already installed, run:
 `sudo apt-get install git`

### CUDA Version Requirement
You must install a CUDA version that is compatible with your GPU and PyTorch version.
Use the official NVIDIA CUDA Toolkit Archive to identify the correct version for your system:
**Note: In this project, we use CUDA version 12.3.0.**
LINK: https://developer.nvidia.com/cuda-12-3-0-download-archive

#### Identifying Your System Configuration

To determine the correct CUDA installer for your Linux system, run the following command in your terminal:
`hostnamectl`
You should see an output similar to the example below:
```
Static hostname: sup2
       Icon name: computer-vm
         Chassis: vm 🖴
      Machine ID: da429e9cb1674e7a8911ea9304f2eb09
         Boot ID: 997fe699749643148736a6a88de11bf6
  Virtualization: google
Operating System: Debian GNU/Linux 12 (bookworm)  
          Kernel: Linux 6.1.0-42-cloud-amd64
    Architecture: x86-64
 Hardware Vendor: Google
  Hardware Model: Google Compute Engine
Firmware Version: Google
```

#### CUDA Installation Selection
Based on this information (operating system, architecture, and Linux distribution), select the appropriate CUDA 12.3.0 installer from the NVIDIA website.
<!-- ![CUDA Installation Diagram](./cuda_installation.png) -->
<img src="./cuda_installation.png" alt="CUDA Installation Diagram" width="90%">

⚠️ **Note:** During the CUDA installation process, under **Installer Type**, copy and use the commands from the first option. In this case, select **`deb (local)`** as shown in the image above.

#### Final Step
Once you have identified the correct CUDA version and installer:
* Copy and paste the code in the `cuda_installation.txt` file in the `unsupervised_wav` folder you cloned from GitHub. 

All commands below should be executed from a terminal.

### Step 1: Make Scripts Executable

```
chmod +x setup_functions.sh \
        wav2vec_functions.sh \
        eval_functions.sh \
        gans_functions.sh\
        run_setup.sh \
        run_wav2vec.sh \
        run_eval.sh \
        run_gans.sh \
        utils.sh
```

### Step 2: Run Environment Setup

This step installs dependencies, configures Fairseq, and prepares the environment.

`./run_setup.sh`


### Step 3: Data Preparation for Unsupervised Wav2Vec-U

**Inputs:**

/path/to/train_audio_dataset – directory of training .wav audio files

/path/to/val_audio_dataset – directory of validation .wav audio files

/path/to/unlabelled/text_dataset – text file containing unlabeled sentences (one per line)

Audio and text inputs are independent and do not require alignment.

⚠️ Note: For the scripts to run successfully:

* All audio files must be in .wav format
* Audio files should have consistent sampling rates (recommended: 16 kHz)

```
./run_wav2vec.sh "/path/to/train_audio_dataset" \
                "/path/to/val_audio_dataset" \
                "/path/to/test_audio_dataset" \
                "/path/to/unlabelled/text_dataset"
```


### Step 4: Configure and GANS Training

Before running run_gans.sh, you may want to adjust the training hyperparameters.

Edit the configuration file:
unsupervised_wav/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu.yaml

You can modify parameters such as:

* Batch size
* Learning rate
* Number of updates

This step is especially useful when working with low-resource datasets.
`./run_gans.sh`

### Step 5: Run Evaluation

During gans training, models are stored in a folder called multirun 

```
the trained checkpoints from train_gans will be stored in a folder called multirun. The checkpoint will be stored in this format 
multirun --
          |
          day/month/year --
                          |
                         time --
                                |
                                checkpoint_best.pt
                                 checkpoint_last.pt
 ```
 Therefore it is advisable to manually provide the path to the exact checkpoint to use under the variable $CHECKPOINT_DIR  in the run_wav2vec.sh script


After training completes, run:

`./run_eval.sh "/path/to/best_gan/model.pt"`
The phone transcription of your test audio will be stored in file called test.txt. The path to the file is below:

```
data --
      |
      transcription_phones --
                            |
                            test.txt
 ```


## GAN Architecture Refactoring

This repository also contains a `gan_refactor` directory, which features a custom encapsulated refactor of the underlying `wav2vec_u.py` Fairseq model.

- **`gan_refactor/refactored_wav2vec_u.py`**: A fully functional, refactored implementation of the Wav2Vec-U model.
  - **Decoupled Logic**: Extracts real phoneme preparation out of the `Generator` and `Wav2vec_U` classes into a distinct `RealData` class.
  - **Single-File Registration**: Remains compatible with the Fairseq registry (`@register_model`) meaning it can be plugged directly into the pipeline without breaking Hydra configs or framework dependencies.

## Summary

1. Install the correct CUDA version for your GPU
2. Make all scripts executable
3. Run `run_setup.sh`
4. Ensure all audio files are `.wav`
5. Optionally adjust GAN training configs
6. Run `run_wav2vec.sh`
7. Run `run_gans.sh`
9. Run `./run_eval.sh "/path/to/best_gan/model.pt"`

