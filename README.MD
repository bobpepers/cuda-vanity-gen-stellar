# Stellar Vanity Address Generator

This is a GPU-accelerated script to generate **Stellar** addresses with a custom **prefix** and **suffix**. It uses **PyCUDA** to leverage your GPU for faster address generation.

## Installing CUDA Toolkit on Ubuntu 22.04

1. **Install the CUDA Toolkit**:
   
   You can install the CUDA toolkit using the following command:

   ```bash
   sudo apt update
   sudo apt install -y nvidia-cuda-toolkit
   ```

## Prerequisites

1. **Python 3.x**: This script requires Python 3. You can check if you have it installed with:

    ```bash
    python3 --version
    ```

2. **NVIDIA GPU with CUDA**: This script uses **PyCUDA** to offload the computation to your GPU. Ensure you have an NVIDIA GPU and CUDA toolkit installed.

    - You can check if CUDA is installed with the following:

    ```bash
    nvcc --version
    ```

3. **Install Dependencies**: The script requires several Python libraries. You can install them using the `requirements.txt` file.

    To install the dependencies, run:

    ```bash
    pip install -r requirements.txt
    ```

## Installation

1. Clone the repository (or download the files):

    ```bash
    git clone https://github.com/bobpepers/vanity-cuda-gen-stellar.git
    cd vanity-gen-stellar
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    venv\Scripts\activate  # For Windows
    ```

3. Install the dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

4. Make sure that you have an NVIDIA GPU and that **CUDA** is set up properly (for PyCUDA to function correctly).

## Usage

1. Run the generator script:

    ```bash
    python3 gen.py
    ```

2. Follow the prompts:
    - **Enter the desired prefix**: Enter the desired start of the Stellar public key.
    - **Enter the desired suffix**: Enter the desired end of the Stellar public key.
    - The script will then attempt to generate keypairs that match the prefix and suffix.
    
    The script will print the **public key** and **secret key** once a match is found.

### Example Output:

Enter the desired prefix: GAME Enter the desired suffix: OVER Searching... 1,000 attempts, Time elapsed: 12.34s Match found after 12,345 attempts! Public Key: GABCDEFGHIJKLMN... Secret Key: SAAAAA1ABCDEF...


## Troubleshooting

- If you get an error like `ModuleNotFoundError: No module named 'numpy'`, make sure you have installed all dependencies using `pip install -r requirements.txt`.
- Ensure that **PyCUDA** is installed correctly. If not, make sure you have **CUDA** installed on your system.
- If you experience issues with the GPU, make sure your **NVIDIA drivers** and **CUDA toolkit** are correctly set up and that your system can access the GPU.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
