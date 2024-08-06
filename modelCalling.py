#imports libraries
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

#sets model_path to the fine-tuned model that was saved from before
model_path = "./SMSModel"

#sets tokenizer and model variable, from the presaved model & tokenizer
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

def is_spam_sms(sms):
    #tokenizes the input sms
    inputs = tokenizer(sms, return_tensors="tf", padding="max_length", truncation=True, max_length=128)

    #passes the tokenized input to the distilbert model
    outputs = model(inputs) #example output: TFSequenceClassifierOutput(loss=None, logits=<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-3.136066,  3.752554]], dtype=float32)>, hidden_states=None, attentions=None)
                                                                                  #logits=<tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[-3.136066,  3.752554]], dtype=float32)>   (not spam,spam)

    #from the outputs recieved by the model, logits is stored in the logits variable (tensorflow library), logits are predictions
    logits = outputs.logits #example output: tf.Tensor([[-3.209868,   3.8740408]], shape=(1, 2), dtype=float32)
    #tf.argmax finds the index of the highest value in logits array
    predicted_class_id = tf.argmax(logits, axis=-1).numpy()[0]
    
    #return Spam if 1, if not return NOT Spam
    return "Spam" if predicted_class_id == 1 else "NOT Spam"

if __name__ == "__main__":
    test_sms = input("Enter a SMS: ")
    result = is_spam_sms(test_sms)
    print(f"Result: {result}")
