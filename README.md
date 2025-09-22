# Federated Learning for Brain Tumor Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style-for-the-badge&logo-python)
![Framework](https://img.shields.io/badge/Flower-1.8-pink?style-for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?style-for-the-badge&logo-tensorflow)

This project is a system for classifying brain tumors from MRI scans using Federated Learning (FL). It uses the Flower framework to train a model across multiple sources without needing to gather the sensitive medical data in one place, which helps protect patient privacy.

---
## Key Features

* **Privacy-Preserving**: Trains on decentralized data, meaning the MRI scans are not moved from their original location.
* **Federated Averaging (FedAvg)**: A custom strategy is used to average the model updates from all clients.
* **Server-Side Evaluation**: The main global model is evaluated on a central test set held by the server. This provides a consistent performance measure and avoids memory issues on client devices.
* **Detailed Logging**: All outputs from the server and clients are saved into log files for review and debugging.
* **Results Persistence**: Automatically saves the final performance metrics from each round to a CSV file.

---
## Tech Stack and Dataset

### Technology Stack

This project combines frameworks for machine learning, federated learning, data manipulation, and image processing.

#### Core Frameworks

* **Python**: The primary programming language used for the project.
    
* **Flower (flwr)**: A framework for building federated learning systems that manages server-client communication.
    
* **TensorFlow**: An end-to-end platform for machine learning, used to build and train the neural network models.
    

#### Data Science and Machine Learning

* **Scikit-learn**: Used for splitting the dataset and calculating performance metrics like precision and F1-score.
    
* **Pandas**: Used to save the final training results into a structured CSV file.
    
* **NumPy**: Essential for handling the multi-dimensional arrays (tensors) that represent images and model weights.
    

#### Image Processing and Utilities

* **Pillow**: A fork of the Python Imaging Library (PIL) that adds image processing capabilities, used by TensorFlow for loading image files.
    
* **OpenCV (opencv-python)**: A library of programming functions mainly aimed at real-time computer vision, useful for advanced image preprocessing.
    
* **imutils**: A series of convenience functions to make basic image processing tasks with OpenCV easier.
* **tqdm**: A library that provides a progress bar for loops, helpful for monitoring long-running tasks.
    

### Dataset: Brain Tumor MRI

This project uses the **Brain Tumor MRI Dataset** from Kaggle, which contains images for four classes: glioma, meningioma, no tumor, and pituitary.

#### How to Download the Dataset

You must use the Kaggle API to download the dataset.

1.  **Install the Kaggle API Client**:

    ```bash
    pip install kaggle
    ```
3.  **Get Your Kaggle API Token**:
    * Go to your Kaggle account page: `https://www.kaggle.com/<Your-Username>/account`.
    * In the "API" section, click **"Create New API Token"** to download your `kaggle.json` file.
4.  **Place the API Token**:
    * Move the downloaded `kaggle.json` file to the `~/.kaggle/` directory (`C:\Users\<Username>\.kaggle` on Windows). You may need to create this folder.
    * For security, make the file readable only by you: `chmod 600 ~/.kaggle/kaggle.json`.
5.  **Download and Unzip the Data**:
    * Run the following command from your project's root directory. It will download and unzip the files into a `data/` folder.
      
    ```bash
    kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p ./data --unzip
    ```

---
## Directory Structure

```
brain-tumor-fl/
├── data/
│   └── Brain-Tumor-MRI-Dataset/
│       ├── Training/
│       └── Testing/
├── logs/
│   ├── client_0.log
│   └── server.log
├── client.py
├── data_loader.py
├── models.py
├── server.py
├── run.sh
└── federated_learning_results.csv
```

---
## Getting Started

### Prerequisites

* Python 3.10 or newer.
* The Brain MRI dataset downloaded and placed in the `data/` folder.

### Installation

1.  **Create a `requirements.txt` file** with this content:
    ```txt
    flwr
    tensorflow
    scikit-learn
    pandas
    numpy
    Pillow
    opencv-python
    imutils
    tqdm
    kaggle
    ```

2.  **Install the required packages:**
   
    ```bash
    pip install -r requirements.txt
    ```

### How to Run

You can start the entire system with the included shell script.

1.  **Run the script:**
   
    ```bash
    ./run.bat
    ```
    This command will create the `logs` directory, start the server, wait a moment, and then start all the clients.

---
## How It Works: Code Breakdown

* **`server.py`**: This is the main control script for the federated system. It initializes the global model, defines the federated averaging strategy, runs the evaluation of the global model on its private test set, starts the Flower server, manages the training rounds, and saves the final metrics after all training is complete.
* **`client.py`**: This script defines what each client does. It loads its specific partition of the training data, creates the machine learning model, receives the global model from the server, trains it on local data, and sends the updated model back.
* **`data_loader.py`**: This is a utility script for all data-related tasks. It loads all images from the dataset directory, splits the data into a training set and a test set, and divides the training set into separate partitions for each client.
* **`models.py`**: This file contains the definitions for the neural network architectures, including a basic CNN and a more advanced model using transfer learning from VGG-16.

---
## Outputs

After the script finishes, you will find these generated files:

1.  **Logs**: The `logs/` directory contains text files with the detailed outputs from the server and each client.
    ```csv
    round,loss,accuracy
    1,1.3981,0.2562
    2,1.2543,0.3451
    ...
    ```
