# chinese-text-emotion-classifier

## 📚 Model Introduction

This model is fine-tuned based on the [joeddav/xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) model, specializing in **Chinese text emotion analysis**.  
Through fine-tuning, the model can identify the following 8 emotion labels:
- **Neutral tone**
- **Concerned tone**
- **Happy tone**
- **Angry tone**
- **Sad tone**
- **Questioning tone**
- **Surprised tone**
- **Disgusted tone**

The model is applicable to various scenarios, such as customer service emotion monitoring, social media analysis, and user feedback classification.

---
# chinese-text-emotion-classifier

## 📚 模型簡介
本模型基於[joeddav/xlm-roberta-large-xnli](https://huggingface.co/joeddav/xlm-roberta-large-xnli) 模型進行微調，專注於 **中文語句情感分析**。  
通過微調，模型可以識別以下 8 種情緒標籤：
- **平淡語氣**
- **關切語調**
- **開心語調**
- **憤怒語調**
- **悲傷語調**
- **疑問語調**
- **驚奇語調**
- **厭惡語調**

該模型適用於多種場景，例如客服情緒監控、社交媒體分析以及用戶反饋分類。

---
## 🚀 Quick Start

### Install Dependencies
Ensure that you have installed Hugging Face's Transformers library and PyTorch:
```bash
pip install transformers torch
```

###Load the Model
Use the following code to load the model and tokenizer, and perform emotion classification:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 添加設備設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 標籤映射字典
label_mapping = {
    0: "平淡語氣",
    1: "關切語調",
    2: "開心語調",
    3: "憤怒語調",
    4: "悲傷語調",
    5: "疑問語調",
    6: "驚奇語調",
    7: "厭惡語調"
}

def predict_emotion(text, model_path="Johnson8187/Chinese-emotion-classifier"):
    # 載入模型和分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)  # 移動模型到設備
    
    # 將文本轉換為模型輸入格式
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)  # 移動輸入到設備
    
    # 進行預測
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 取得預測結果
    predicted_class = torch.argmax(outputs.logits).item()
    predicted_emotion = label_mapping[predicted_class]
    
    return predicted_emotion

if __name__ == "__main__":
    # 使用範例
    test_texts = [
        "雖然我努力了很久，但似乎總是做不到，我感到自己一無是處。",
        "你說的那些話真的讓我很困惑，完全不知道該怎麼反應。",
        "這世界真的是無情，為什麼每次都要給我這樣的考驗？",
        "有時候，我只希望能有一點安靜，不要再聽到這些無聊的話題。",
        "每次想起那段過去，我的心還是會痛，真的無法釋懷。",
        "我從來沒有想過會有這麼大的改變，現在我覺得自己完全失控了。",
        "我完全沒想到你會這麼做，這讓我驚訝到無法言喻。",
        "我知道我應該更堅強，但有些時候，這種情緒真的讓我快要崩潰了。"
    ]

    for text in test_texts:
        emotion = predict_emotion(text)
        print(f"文本: {text}")
        print(f"預測情緒: {emotion}\n")

```

---
## 🚀 快速開始

### 安裝依賴
請確保安裝了 Hugging Face 的 Transformers 庫和 PyTorch：
```bash
pip install transformers torch
```

### 加載模型
使用以下代碼加載模型和分詞器，並進行情感分類：
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 添加設備設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 標籤映射字典
label_mapping = {
    0: "平淡語氣",
    1: "關切語調",
    2: "開心語調",
    3: "憤怒語調",
    4: "悲傷語調",
    5: "疑問語調",
    6: "驚奇語調",
    7: "厭惡語調"
}

def predict_emotion(text, model_path="Johnson8187/Chinese-emotion-classifier"):
    # 載入模型和分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)  # 移動模型到設備
    
    # 將文本轉換為模型輸入格式
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)  # 移動輸入到設備
    
    # 進行預測
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 取得預測結果
    predicted_class = torch.argmax(outputs.logits).item()
    predicted_emotion = label_mapping[predicted_class]
    
    return predicted_emotion

if __name__ == "__main__":
    # 使用範例
    test_texts = [
        "雖然我努力了很久，但似乎總是做不到，我感到自己一無是處。",
        "你說的那些話真的讓我很困惑，完全不知道該怎麼反應。",
        "這世界真的是無情，為什麼每次都要給我這樣的考驗？",
        "有時候，我只希望能有一點安靜，不要再聽到這些無聊的話題。",
        "每次想起那段過去，我的心還是會痛，真的無法釋懷。",
        "我從來沒有想過會有這麼大的改變，現在我覺得自己完全失控了。",
        "我完全沒想到你會這麼做，這讓我驚訝到無法言喻。",
        "我知道我應該更堅強，但有些時候，這種情緒真的讓我快要崩潰了。"
    ]

    for text in test_texts:
        emotion = predict_emotion(text)
        print(f"文本: {text}")
        print(f"預測情緒: {emotion}\n")

```

---

### Dataset
- The fine-tuning dataset consists of 4,000 annotated Traditional Chinese emotion samples, covering various emotion categories to ensure the model's generalization capability in emotion classification.
- [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)


### 數據集
- 微調數據來自4000個自行標註的高質量繁體中文情感語句數據，覆蓋了多種情緒類別，確保模型在情感分類上的泛化能力。
- [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

---

🌟 Contact and Feedback
If you encounter any issues while using this model, please contact:

Email: fable8043@gmail.com
Hugging Face Project Page: chinese-text-emotion-classifier

## 🌟 聯繫與反饋
如果您在使用該模型時有任何問題，請聯繫：
- 郵箱：`fable8043@gmail.com`
- Hugging Face 項目頁面：[chinese-text-emotion-classifier](https://huggingface.co/Johnson8187/chinese-text-emotion-classifier)
