# AI Playground using [HuggingFace](https://huggingface.co/)  

* [**HuggingFace Documentation**](https://huggingface.co/docs)  
  * [**Hub**](https://huggingface.co/docs/hub/index)  
    * [**Hub client library**](https://huggingface.co/docs/huggingface_hub/index)  
  * [**Transformers**](https://huggingface.co/docs/transformers/index)  
  * [**Diffusers**](https://huggingface.co/docs/diffusers/index)  
  * [**Accelerate**](https://huggingface.co/docs/accelerate/index)  
  * [**Safetensors**](https://huggingface.co/docs/safetensors/index)  


## How To Run

* To run experiments, with _text-to-image using a provider_ as an example:
  1. `cd` into project's main directory:  
     `cd <...>/ai_HuggingFace/hf_provider/text-to-image`  
  3. export the access token environment variable:  
     `export "HF_TOKEN"=<value of token>`  
  4. `./predict.sh`

* The YAML files for creating the conda environments that are used in experiments are included in the `conda` directory. 

## Experiments: text-to-image using a provider  

* The model used is [**black-forest-labs/FLUX.1-dev**](https://huggingface.co/black-forest-labs/FLUX.1-dev)  
* The prompt used as input is as follows:  
  **Minimalist ink line drawing, loose gestural brushstrokes, black ink on cream/beige background, capture essential contours with minimal lines, confident flowing varied thickness, omit unnecessary details, intimate and contemplative mood, sumi-e + modern line art style, high contrast, use negative space effectively, create an impression rather than an exact likenes.**

|     |
|:---:|
![][fig_1]

[fig_1]:hf_provider/text-to-image/predictions/prompt.png

