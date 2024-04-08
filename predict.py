from transformers import BertForSequenceClassification, BertTokenizer
import torch
import gradio as gr
# 载入tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 加载训练好的模型
model = BertForSequenceClassification.from_pretrained(r"D:\huggingface\中文预训练\saved_models\bert_model\bert_model_epoch_250")

# 定义预测函数
# 定义标签映射
label_map = {0: "negative", 1: "neutral", 2: "positive"}

# 定义预测函数
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits).item()
    predicted_label = label_map[predicted_class]
    return predicted_label


# 启动 Gradio 接口
iface = gr.Interface(fn=predict_sentiment, inputs="text", outputs="label", title="Sentiment Analysis")
iface.launch()
