# Import necessary libraries
import pandas as pd
import numpy as np
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModel

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set file paths for train and test data
train_path = "/content/drive/MyDrive/dataset/train.csv"
test_path = "/content/drive/MyDrive/dataset/test.csv"

# Load train and test data
df_train = pd.read_csv(train_path, escapechar="\\", quoting=csv.QUOTE_NONE, error_bad_lines=False)
df_test = pd.read_csv(test_path, escapechar="\\", quoting=csv.QUOTE_NONE, error_bad_lines=False)

# Drop rows with missing values from train data
df_train = df_train.dropna()

# Preprocess train and test data
punctuation_signs = list("?:!.,;")
nltk.download('punkt')
nltk.download('wordnet')
wordnet_lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))

df_train['TITLE'] = df_train['TITLE'].str.replace("\r", " ")
df_train['TITLE'] = df_train['TITLE'].str.replace("\n", " ")
df_train['TITLE'] = df_train['TITLE'].str.replace("    ", " ")
df_train['TITLE'] = df_train['TITLE'].str.replace('"', '')
df_train['TITLE'] = df_train['TITLE'].str.lower()
for punct_sign in punctuation_signs:
  df_train['TITLE'] = df_train['TITLE'].str.replace(punct_sign, '')
df_train['TITLE'] = df_train['TITLE'].str.replace("'s", "")

final_cols = ["TITLE", "PRODUCT_LENGTH"]
df_train = df_train[final_cols]
df_train = df_train.iloc[:35000, :]

df_test['TITLE'] = df_test['TITLE'].str.replace("\r", " ")
df_test['TITLE'] = df_test['TITLE'].str.replace("\n", " ")
df_test['TITLE'] = df_test['TITLE'].str.replace("    ", " ")
df_test['TITLE'] = df_test['TITLE'].str.replace('"', '')
df_test['TITLE'] = df_test['TITLE'].str.lower()
for punct_sign in punctuation_signs:
  df_test['TITLE'] = df_test['TITLE'].str.replace(punct_sign, '')
df_test['TITLE'] = df_test['TITLE'].str.replace("'s", "")

final_cols = ["TITLE", "PRODUCT_ID"]
df_test = df_test[final_cols]

# Fill missing values in test data
df_test["TITLE"].fillna("No Data", inplace = True)

# Split train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(df_train['TITLE'], df_train['PRODUCT_LENGTH'], test_size=0.2, random_state=42)

# Instantiate BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize train and test data
X_train_tokens = tokenizer.batch_encode_plus(X_train.tolist(),max_length = 25,pad_to_max_length=True,
truncation=True,
return_token_type_ids=False)
X_val_tokens = tokenizer.batch_encode_plus(X_val.tolist(),
max_length = 25,
pad_to_max_length=True,
truncation=True,
return_token_type_ids=False)

X_test_tokens = tokenizer.batch_encode_plus(df_test['TITLE'].tolist(),
max_length = 25,
pad_to_max_length=True,
truncation=True,
return_token_type_ids=False)

# Convert tokens to tensors
X_train_seq = torch.tensor(X_train_tokens['input_ids'])
X_train_mask = torch.tensor(X_train_tokens['attention_mask'])

X_val_seq = torch.tensor(X_val_tokens['input_ids'])
X_val_mask = torch.tensor(X_val_tokens['attention_mask'])

X_test_seq = torch.tensor(X_test_tokens['input_ids'])
X_test_mask = torch.tensor(X_test_tokens['attention_mask'])

# Set batch size
batch_size = 32

# Create train DataLoader
train_data = TensorDataset(X_train_seq, X_train_mask, torch.tensor(y_train.values))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create validation DataLoader
val_data = TensorDataset(X_val_seq, X_val_mask, torch.tensor(y_val.values))
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Create test DataLoader
test_data = TensorDataset(X_test_seq, X_test_mask)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Define BERT regression model
class BERTRegression(nn.Module):
def init(self):
super(BERTRegression, self).init()
self.bert = model
self.dropout = nn.Dropout(0.2)
self.linear = nn.Linear(768, 1)

def forward(self, seq, mask):
_, pooled_output = self.bert(seq, attention_mask=mask, return_dict=False)
x = self.dropout(pooled_output)
x = self.linear(x)
return x

Instantiate BERT regression model
model = BERTRegression()
model.cuda()

Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
num_warmup_steps = 0,
num_training_steps = total_steps)

Define mean absolute percentage error loss function
def mape_loss(actual, predicted):
return torch.mean(torch.abs((actual - predicted) / actual)) * 100

Train BERT regression model
for epoch in range(epochs):
model.train()
total_loss = 0
for step, batch in enumerate(train_dataloader):
batch_seq = batch[0].to(device)
batch_mask = batch[1].to(device)
batch_labels = batch[2].to(device)
model.zero_grad()
outputs = model(batch_seq, batch_mask)
loss = mape_loss(outputs.squeeze(), batch_labels.float())
total_loss += loss.item()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
avg_train_loss = total_loss / len(train_dataloader)
model.eval()
eval_loss = 0
for batch in val_dataloader:
batch_seq = batch[0].to(device)
batch_mask = batch[1].to(device)
batch_labels = batch[2].to(device)
with torch.no_grad():
outputs = model(batch_seq, batch_mask)

Extract features from train and validation sets
train_features = tokenizer(X_train.tolist(), padding=True, truncation=True)
val_features = tokenizer(X_val.tolist(), padding=True, truncation=True)

# Convert features to tensors
train_input_ids = torch.tensor(train_features['input_ids'])
train_attention_mask = torch.tensor(train_features['attention_mask'])
val_input_ids = torch.tensor(val_features['input_ids'])
val_attention_mask = torch.tensor(val_features['attention_mask'])
train_labels = torch.tensor(y_train.values)
val_labels = torch.tensor(y_val.values)

# Define logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Train logistic regression model
logreg.fit(train_input_ids, train_labels)

# Make predictions on validation set
val_predictions = logreg.predict(val_input_ids)

# Evaluate performance on validation set
print("Accuracy:", accuracy_score(val_labels, val_predictions))
print("F1 score:", f1_score(val_labels, val_predictions, average='weighted'))

# Extract features from test set
test_features = tokenizer(df_test['TITLE'].tolist(), padding=True, truncation=True)

# Convert features to tensors
test_input_ids = torch.tensor(test_features['input_ids'])
test_attention_mask = torch.tensor(test_features['attention_mask'])

Make predictions on test set
test_predictions = logreg.predict(test_input_ids)

Save predictions to submission file
submission_df = pd.DataFrame({'PRODUCT_ID': df_test['PRODUCT_ID'], 'PRODUCT_LENGTH': test_predictions})
submission_df.to_csv('/content/drive/MyDrive/submission.csv', index=False)