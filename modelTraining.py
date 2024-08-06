#imports libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
import tensorflow as tf

#reads the SMSSpamCollection file, separates it based on space (\t). the first column is called labels, second is called message
df = pd.read_csv('C:\\Prajit\\AI\\modelTraining\\SMSSpamCollection.csv', sep='\t', names=["label", "message"])

#using the message column from earlier, organizes into list
X = list(df['message'])

#using the label column from earlier, organizes into list
y = list(df['label'])

#sorts the file in alphabetical order, makes it binary, ham=0, spam=1
#y = list(pd.get_dummies(y, drop_first=True)['spam'])

#creates four variables and splits the dataset randomly among them, 20% of each column goes to the test variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

#loads in the tokenizer from the distilbert model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#tokenizes the x_train and x_test variables
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)
y_test = tokenizer(y_test, truncation=True, padding=True)
y_train = tokenizer(y_train, truncation=True, padding=True)

#creates train_dataset which is what the model is trained on
#shuffles the dataset in random order, groups data in groups of 8
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train)).shuffle(1000).batch(8)

#creates test_dataset to evaluate model's performance
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test)).batch(16)

#initialize the distilbert model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

#sets default learning rate and using adam it adapts during training
#learning rate determines how much to adjust the models' prediction (weights)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

#determines how the model's output is by comparing it with the test data
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#used to evaluate model accuracy 
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

#prepares the model for training by passing on the above variables to compile function
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

#trains the model on the dataset, goes through entire dataset 2 times, test_dataset is used to evaluate model's performance
model.fit(train_dataset, epochs=2, validation_data=test_dataset)

#saves model and tokenizer (each model has unique tokenizer)
model.save_pretrained("./SMSModel")
tokenizer.save_pretrained("./SMSModel")
