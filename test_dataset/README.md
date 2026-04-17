## 🐬 Introduction

This test dataset contains **1,080** foundational test samples, addressing a multitude of classification tasks, including generic safety classification, multilingual risk identification, adversarial attack defense, and safe completion. Furthermore, the dataset supports diverse input scenarios, accommodating both general text content (user prompts or model responses) and interconnected query-response dialogue pairs.


## ⚙️ Dataset Details

### Field Descriptions

Each instance in the dataset is structured with the following fields:
| Field | Description |
| :--- | :--- |
| **`id`** | A unique identifier for the data sample. |
| **`prompt`** | The original user query. |
| **`response`** | The model-generated reply. |
| **`stage`** | Indicates the input scenario of the sample. It takes one of three values: <br>- `q`: Input the prompt only, where the `response` field is empty; <br>- `r`: Input the response only, where the `prompt` field is empty; <br>- `qr`: Input the interaction, where `prompt` and `response` fields are not empty. |
| **`label`** | The risk tag (`safe` \ `unsafe`) of the test sample, depending on the current stage: <br>- `stage = q`: Risk tag of `prompt`; <br>- `stage = r`: Risk tag of `response`; <br>- `stage = qr`: Risk tag of the interaction. |


## ⚠️ Disclaimer

This dataset is intended to facilitate the establishment of a security governance framework for large models and to accelerate their safe and controllable application. It may contain offensive or upsetting content, and is provided solely for research and lawful purposes. The views expressed in the data are not related to the organizations, authors and affiliated entities involved in this project. Users assume full responsibility for the final behavior and compliance of downstream systems. This project is not liable for any direct or indirect losses resulting from the use of this dataset.


## 📄 License

This project is licensed under the Apache-2.0 License.