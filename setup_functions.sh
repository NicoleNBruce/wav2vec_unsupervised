#!/bin/bash

# This script holds all the functions and libraries needed a successful setup 
# that ensure a smooth running of the fairseq wav2vec unsupervised pipeline

set -e                       # Exit on error
set -o pipefail              # Exit if any command in a pipe fails
set -x                       # Print each command for debugging

# ==================== CONFIGURATION ====================
# Set these variables according to your environment

# Main directories
INSTALL_ROOT="$HOME/wav2vec_unsupervised"
FAIRSEQ_ROOT="$INSTALL_ROOT/fairseq_"
KENLM_ROOT="$INSTALL_ROOT/kenlm"
VENV_PATH="$INSTALL_ROOT/venv"
RVADFAST_ROOT="$INSTALL_ROOT/rVADfast"
FLASHLIGHT_SEQ_ROOT="$INSTALL_ROOT/sequence"


# Python version
PYTHON_VERSION="3.10"  # Options: 3.7, 3.8, 3.9, 3.10
CUDA="12.3"


# ==================== HELPER FUNCTIONS ====================

# Log message with timestamp
log() {
    local message="$1"
    local timestamp
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $message"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

get_system_cuda_suffix() {
    if ! command -v nvcc --version >/dev/null 2>&1; then
        log "ERROR: nvcc (NVIDIA CUDA Compiler) not found in PATH. Cannot determine CUDA version for GPU packages."
        exit 1
    fi
    local cuda_version
    cuda_version=$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
}

# Create home and log directory 
create_dirs() {
    mkdir -p "$INSTALL_ROOT"
    mkdir -p "$INSTALL_ROOT/logs"
}


# ==================== SETUP STEPS ====================
setup_venv() {
    log "Setting up Python virtual environment..."
    
     #setting up pyenv to tackle linkage errors, protobuf requires a python environment which is not static 
    export PYENV_ROOT="$HOME/.pyenv"

    # Install pyenv ONLY if not already installed
    if [ ! -d "$PYENV_ROOT" ]; then
        curl -fsSL https://pyenv.run | bash
            export PYENV_ROOT="$HOME/.pyenv"
            [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
            eval "$(pyenv init - bash)"
        echo "Detected Python version: $PYTHON_VERSION"
        env PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install $PYTHON_VERSION
        pyenv local $PYTHON_VERSION
    else
        log "Python $PYENV_ROOT already installed."
    fi
   
    
    if [ -d "$VENV_PATH" ]; then
        log "Virtual environment already exists at $VENV_PATH"
    else
        # python${PYTHON_VERSION} -m venv "$VENV_PATH"
        python3 -m venv "$VENV_PATH"
        log "Created virtual environment at $VENV_PATH"
    fi
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"

    # Set global pip timeouts to prevent Mac-to-Docker network dropouts
    export PIP_DEFAULT_TIMEOUT=1000
    pip config set global.timeout 1000

    log "Python virtual environment setup completed."
}

#installing_python_basic_dependencies
basic_dependencies(){
    sudo apt-get update
    # Install Python 3, pip, and essential development packages (for compiling C extensions)
    sudo apt-get install -y python3 python3-pip python3-dev build-essential 
    sudo apt-get install autoconf automake cmake curl g++ git graphviz libatlas3-base libtool make pkg-config subversion unzip wget zlib1g-dev gfortran
    sudo apt update
    sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev wget curl
    sudo apt install software-properties-common

}

# This function installs the cuda version suited for your machine

cuda_installation() {
    local cmd_file="cuda_installation.txt"

    if [[ -f "$cmd_file" ]]; then
        echo "Starting installation from $cmd_file..."

        source "$cmd_file"

        # Add CUDA to PATH safely (without expanding PATH immediately)
        echo 'export PATH=/usr/local/cuda-'"$CUDA"'/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-'"$CUDA"'/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

        # Apply immediately to current shell
        export PATH="/usr/local/cuda-$CUDA/bin:$PATH"
        export LD_LIBRARY_PATH="/usr/local/cuda-$CUDA/lib64:$LD_LIBRARY_PATH"

        source ~/.bashrc

        echo "CUDA environment variables configured."
    else
        echo "Error: $cmd_file not found!"
        return 1
    fi
}


#Installation of gpu drivers and toolkit
gpu_drivers_installation(){
 echo "--- Starting GPU Driver and Toolkit Installation ---"
    # 1. Download Google Cloud GPU installation script
    echo "1. Downloading GCP GPU driver installation script..."
    curl -s -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py

    # 2. Run the installation script
    echo "2. Running GPU driver installation script"
    sudo python3 install_gpu_driver.py

    # 3. Update package lists
    echo "3. Updating package lists..."
    sudo apt-get update -y

    # --- 4. Verify nvidia-smi and Fix PATH if necessary ---
    echo "4. Verifying installation location and fixing PATH..."

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        echo "=================================================================="
        echo "SUCCESS: 'nvidia-smi' is now found and running the command:"
        nvidia-smi
        echo "=================================================================="
    else
        echo ""
        echo "=================================================================="
        echo "FAILURE: 'nvidia-smi' is not found in the initial system PATH."
        
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "SUCCESS: PATH fix worked! Running the command:"
            nvidia-smi
            echo "=================================================================="
        else
            echo "CRITICAL FAILURE: Driver utilities are missing or severely misconfigured."
            echo "A system reboot may be required to fully activate the newly installed driver."
            echo "=================================================================="
        fi
    fi
}

# Install PyTorch and related packages
install_pytorch_and_other_packages() {
    log "Installing PyTorch and related packages..."
    source "$VENV_PATH/bin/activate"
  
    pip install --upgrade pip
    
    # pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/cu121"
    # pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/cpu"
    pip install --default-timeout=1000 torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url "https://download.pytorch.org/whl/cpu"


    # Install other required packages
    pip install "numpy<2" scipy tqdm sentencepiece soundfile librosa editdistance tensorboardX packaging soundfile
    pip install npy-append-array h5py kaldi-io g2p_en

    if ! command -v nvcc --version >/dev/null 2>&1; then
         pip install faiss-cpu
    else
        pip install faiss-gpu
    fi
    
    # pip install ninja
    # pip install torchcodec
    # sudo apt install zsh

    pip install --default-timeout=1000 ninja torchcodec    
    sudo apt-get install -y zsh
    pip install --default-timeout=1000 nltk
    python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')" # we install this to efficiently use the phonemizer G2p

    log "PyTorch and related packages installed successfully."
}

# Clone and install fairseq
install_fairseq() {
    log "--- Installing fairseq ---"
    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
     pip install "pip==24.0"
    # source "$VENV_PATH/bin/activate"

    cd "$INSTALL_ROOT"

    if [ -d "$FAIRSEQ_ROOT" ]; then
        log "fairseq repository already exists. Pulling latest changes..."
        cd "$FAIRSEQ_ROOT"
        git pull || { log "[WARN] Failed to pull latest fairseq changes. Continuing with existing version."; }
    else
        log "Cloning fairseq repository..."
        # git clone https://github.com/facebookresearch/fairseq.git "$FAIRSEQ_ROOT" \
        git clone https://github.com/Ashesi-Org/fairseq_.git "$FAIRSEQ_ROOT" || { log "[ERROR] Failed to clone fairseq repository."; exit 1; }
        cd "$FAIRSEQ_ROOT"
    fi

    # --- INJECT CUSTOM REFACTORED WAV2VEC_U ARCHITECTURE BEFORE INSTALL ---
    log "Injecting custom refactored Wav2Vec-U architecture..."
    cp "$SCRIPT_ROOT/gan_refactor/refactored_wav2vec_u.py" "$FAIRSEQ_ROOT/fairseq/models/wav2vec/wav2vec_u.py"

    log "Installing fairseq in editable mode..."
    pip install --editable ./ \
        || { log "[ERROR] Failed to install fairseq in editable mode."; exit 1; }

    # Install wav2vec specific requirements if the file exists
    local wav2vec_req_file="$FAIRSEQ_ROOT/examples/wav2vec/requirements.txt"
    if [ -f "$wav2vec_req_file" ]; then
        log "Installing wav2vec specific requirements from $wav2vec_req_file..."
        pip install -r "$wav2vec_req_file" \
            || { log "[WARN] Failed to install some wav2vec requirements. Check $wav2vec_req_file."; }
    else
        log "[INFO] No specific requirements file found at $wav2vec_req_file."
    fi

    log "fairseq installed successfully."
    deactivate
}


#Install rVADfast for audio silence removal
install_rVADfast() {
    log "Cloning and installing rVADfast..."
    cd "$INSTALL_ROOT"
    
    source "$VENV_PATH/bin/activate"

    if [ -d "$RVADFAST_ROOT" ]; then
        log "rVADfast already exists. Updating..."
        cd "$RVADFAST_ROOT"
        git pull
    else
        log "Cloning rVADfast repository..."
        git clone https://github.com/zhenghuatan/rVADfast.git "$RVADFAST_ROOT"
        cd "$RVADFAST_ROOT"
    fi

    mkdir -p "$RVADFAST_ROOT/src"
    
    log "rVADfast installed successfully."
}

#  Clone and build KenLM
install_kenlm() {
    log "Cloning and building KenLM..."
    cd "$INSTALL_ROOT"

    sudo apt update
    sudo apt install libeigen3-dev

    sudo apt update
    sudo apt install libboost-all-dev

    if [ -d "$KENLM_ROOT" ]; then
        log "KenLM repository already exists."
    else
        log "Cloning KenLM repository..."
        git clone https://github.com/kpu/kenlm.git "$KENLM_ROOT"
    fi
    
    cd "$KENLM_ROOT"
    if [ -d "build" ]; then
        log "KenLM build directory already exists. Skipping build step."
    else  
        mkdir -p build
        cd build
        cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        make -j $(nproc)
    fi
    
    source "$VENV_PATH/bin/activate"
    pip install https://github.com/kpu/kenlm/archive/master.zip
    
    log "KenLM built successfully."
}

#  Install Flashlight and Flashlight-Sequence
install_flashlight() {
    log "--- Installing Flashlight (Text and Sequence) ---"
    cd "$INSTALL_ROOT"

    sudo apt-get install pybind11-dev

    # Ensure  nvcc is installed to before proceeding with GPU build
    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"

    # Install flashlight-text (Python-only package)
    log "Installing flashlight-text Python package..."
    pip install flashlight-text \
        || { log "[ERROR] Failed to install flashlight-text."; exit 1; }

    # Clone or update the sequence repository
    if [ -d "$FLASHLIGHT_SEQ_ROOT" ]; then
        log "Flashlight sequence repository already exists. Updating..."
        cd "$FLASHLIGHT_SEQ_ROOT"
        git pull || { log "[WARN] Failed to pull latest flashlight sequence changes."; }
    else
        log "Cloning flashlight sequence repository..."
        git clone https://github.com/flashlight/sequence.git "$FLASHLIGHT_SEQ_ROOT" \
            || { log "[ERROR] Failed to clone flashlight sequence."; exit 1; }
        cd "$FLASHLIGHT_SEQ_ROOT"
    fi

    log "Configuring and Building flashlight sequence library WITH Python bindings..."
    # Remove old build directory for a clean state
    rm -rf build
    mkdir build && cd build

 
    local flashlight_python_flag="-DFLASHLIGHT_BUILD_PYTHON=ON" # <--- CHECK THIS FLAG!
    log "[INFO] Using CMake flag for Python bindings: $flashlight_python_flag (Verify this is correct!)"

    export USE_CUDA=1 # Set if building for CUDA

    if ! command -v nvcc &> /dev/null; then
        log "[INFO] nvcc not found. Switching to CPU-only build."
        use_cuda_flag="-DFLASHLIGHT_USE_CUDA=OFF"
        export USE_CUDA=0
    fi

    # Explicitly point CMake to the Python executable in the venv for robustness
    local python_executable="$VENV_PATH/bin/python"
    cmake .. -DCMAKE_BUILD_TYPE=Release \
             -DPYTHON_EXECUTABLE="$python_executable" \
             "$flashlight_python_flag" \
             "$use_cuda_flag" 

    # Build the C++ library AND Python bindings
    log "Building Flashlight sequence (C++ and Python)..."
    cmake --build . --config Release --parallel "$(nproc)" 
     
    # Install the Python Bindings into the ACTIVE virtual environment
    log "Installing Flashlight sequence Python bindings into venv..."
    # This assumes setup.py or similar is generated in the build directory.
    cd ..
    pip install . 

    log "[PASS] Flashlight Python bindings installed via pip."

    cd "$INSTALL_ROOT" # Go back to install root
    log "Flashlight installation steps completed."

    # --- Re-install fairseq AFTER Flashlight bindings are in venv ---
    log "Re-installing fairseq to ensure it picks up Flashlight bindings..."
    install_fairseq # Call the fairseq install function again (it will activate/deactivate venv)

    log "--- Flashlight Installation Finished ---"
    # Final deactivate handled by install_fairseq
}

#  Download pre-trained wav2vec model
download_pretrained_model() {
    log "Downloading pre-trained wav2vec model..."
    
    mkdir -p "$INSTALL_ROOT/pre-trained"
    cd "$INSTALL_ROOT/pre-trained"
    
    if [ -f "$INSTALL_ROOT/pre-trained/wav2vec_vox_new.pt" ]; then
        log "Pre-trained model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt
    fi
    
    log "Pre-trained model downloaded successfully."
}

# Download language identification model
download_languageIdentification_model() {
    log "Downloading language identification model..."
    
    mkdir -p "$INSTALL_ROOT/lid_model"
    cd "$INSTALL_ROOT/lid_model"
    
    if [ -f "$INSTALL_ROOT/lid_model/lid.176.bin" ]; then
        log "LID model already exists. Skipping download."
    else
        wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
    fi

    source "$VENV_PATH/bin/activate"
    pip install fasttext
    
    log "Language identification model downloaded successfully."
}

