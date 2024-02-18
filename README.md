# Fine-Tuning CodeLlama-7b-Instruct-hf Model
## 1. Model Used

In this project we are using the CodeLlma -7b-Instruct-hf model which is basically used for generating the code.
We take the base model [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) from the huggingface hub.

## 2. Fine-Tuning Process

Fine-tuning is the process of taking a pretrained model and adapting it to perform specific tasks or solve particular problems. In this project, the fine-tuning process involves several critical steps:

### 2.1. Tokenization

We use the `AutoTokenizer` from the Hugging Face Transformers library to tokenize the base model. This step prepares the model for training on specific tasks by converting text data into a suitable format.

### 2.2. Quantization

Quantization is applied to the base model using a custom configuration. This process optimizes the model for efficient execution while minimizing memory usage. We employ the following quantization parameters:

- `load_in_4bit`: Activates 4-bit precision for base model loading.
- `bnb_4bit_use_double_quant`: Uses double quantization for 4-bit precision.
- `bnb_4bit_quant_type`: Specifies the quantization type as "nf4" (Nested float 4-bit).
- `bnb_4bit_compute_dtype`: Sets the compute data type to torch.bfloat16.

### 2.3. LoRA (Low-Rank Adaptation) Configuration

LoRA (Low Rank Adaptation) is a new technique for fine-tuning deep learning models that works by reducing the number of trainable parameters. Key parameters for LoRA include:

- `lora_r`: LoRA attention dimension set to 8.
- `lora_alpha`: Alpha parameter for LoRA scaling set to 16.
- `lora_dropout`: Dropout probability for LoRA layers set to 0.05.

### 2.4. Training Configuration

We configure various training parameters, including batch sizes, learning rates, and gradient accumulation steps. Some of the key training parameters are:

- Batch size per GPU for training and evaluation
- Gradient accumulation steps
- Maximum gradient norm (gradient clipping)
- Initial learning rate (AdamW optimizer)
- Weight decay
- Optimizer type (e.g., paged_adamw_8bit)
- Learning rate schedule (e.g., cosine)

### 2.5. Supervised Fine-Tuning (SFT)

We employ a Supervised Fine-Tuning (SFT) approach to train the model on specific tasks. This involves providing labeled datasets related to the tasks LLM should specialize in.

### 2.6. Model Saving

After training, the specialized models are saved for future use.

## 3. Fine-Tuning Processes

The fine-tuning process consists of several key steps:

- Tokenization: Transforming text data into a format suitable for the model.
- Quantization: Optimizing the model for efficiency and memory usage.
- LoRA Configuration: Reduce the number of trainable parameters.
- Training Configuration: Setting up training parameters and optimizations.
- Supervised Fine-Tuning (SFT): Training the model on specific tasks using labeled data.
- Model Saving: Saving the trained models for later use.

## 4. GPU Requirements

The fine-tuning process is computationally intensive and requires a GPU with sufficient capabilities to handle the workload effectively. While the specific GPU requirements may vary depending on the size of the model and the complexity of the tasks, it is recommended to use a high-performance GPU with CUDA support. Additionally, the availability of VRAM (Video RAM) is crucial, as large models like `codellama/CodeLlama-7b-Instruct-hf` can consume significant memory during training. 

In this project, we have set the `device` to use CUDA, so we are using the google colab 15GB T4 GPU for fine-tuning.

Please ensure that your GPU meets the necessary hardware and software requirements to successfully execute the fine-tuning process.

## Usage

### 1. Colab Notebook (recommended)

This is the simplest and easiest way to run this project.

1. Locate the `Fine-tuning-CodeLlama_demo.ipynb` in this repo
2. Click the "Open in Colab" button at the top of the file
3. Change the runtime type to T4 GPU
4. Run all the cells in the notebook

### 2. Run Locally

Inferencing this model locally requires a GPU with atleast 16GB of GPU RAM.

#### Instructions:

1. Clone this repository to your local machine.
```bash
git clone https://github.com/VivekChauhan05/Fine-tuning_CodeLlama-7b.git
```

2. Navigate to project directory.
```bash
cd Fine-tuning_CodeLlama-7b_Model
```

3. Install the required dependencies. 
```bash
pip install -r requirements.txt
```

4. Run the `app.py` file.
```bash
python app.py
```

5. Open the link provided in the terminal in your browser.

For more details on the code implementation and usage, refer to the code files in this repository.


## License

This project is licensed under [Apache 2.0](LICENSE)