# LivingEBG_EvidenceSurveillance_PCOS2023

## Requirements 

### System Requirements
* Python version >=3.11, <3.13
* Git LFS (for handling large files)

### Supported Platforms 
* Windows 10
* Ubuntu 22.04 (Tested on WSL)

### GPU Support (Optional)
* CUDA 12.0 or higher (only needed for GPU environment)

## Installation 

1. **Install Prerequisites**

### GIT LFS 

   ```bash
   # Install Git LFS
   git lfs install
   ```
### Pixi package manager 

* Install Pixi: https://pixi.sh/latest/ 

2. **Clone Repository**
   ```bash
   git clone https://github.com/your-username/LivingEBG_EvidenceSurveillance_PCOS2023
   cd LivingEBG_EvidenceSurveillance_PCOS2023
   git lfs pull  # Pull large files
   ```

3. **Choose Environment**
   - For systems with CUDA-enabled GPU:
     ```bash
     pixi install --environment gpu
     ```
   - For CPU-only systems:
     ```bash
     pixi install --environment cpu
     ```

4. **Activate Environment**
   ```bash
   pixi shell
   ```

## Usage


1. **Verify Environment for Torch**
   ```bash
   pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

2. Run the following scripts to replicate the results in the paper: 

* Run main_overarching.py to regenerate overarching search results and evaluation metrics 


