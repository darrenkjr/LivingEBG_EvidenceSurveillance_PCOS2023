# LivingEBG_EvidenceSurveillance_PCOS2023

## Requirements 

### System Requirements
* Python version >=3.11, <3.13
* Git >=2.0, though latest version is recommended
* Git LFS (for handling large files)

### Supported Platforms (Tested) 
* Windows 10
* Ubuntu 22.04 (Tested via WSL2)
* Rocky Linux 9.2 

### GPU Support (Optional)
* CUDA 12.0 or higher (only needed for GPU environment)

## Installation 

1. **Install Prerequisites**

### GIT LFS 

For Windows and most cases: 
* Follow GIT LFS installation instruction available here: https://git-lfs.com/

In instances with no sudo access, or you would simply like to install in your user account without accessing the server root account: 
* Follow the instructions here: https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab 

### Pixi package manager 

* Install and follow instructions for Pixi: https://pixi.sh/latest/ 

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

2. Setup environment variables

* Copy .env_example and rename it to .env, then replace the variables with your own values (see comments in .env_example)

3. Run the following scripts to replicate the results in the paper: 

* Run main_overarching.py to regenerate overarching search results and evaluation metrics 


