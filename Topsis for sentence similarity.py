#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install datasets


# In[4]:


pip install sentence-transformers


# In[3]:


from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset


# In[4]:


dataset = load_dataset("mteb/stsbenchmark-sts")


# In[5]:


dataset


# In[7]:


train_data=dataset['train']


# In[8]:


train_data


# In[9]:


import pandas as pd


# In[10]:


train_df=pd.DataFrame(train_data)


# In[11]:


train_df.head()


# In[12]:


train_df=train_df.iloc[:,5:]


# In[13]:


train_df.head()


# In[17]:


# Define the new range
new_min_range = -1
new_max_range = 1 

# Calculate the original min and max values
original_min_value = train_df['score'].min()
original_max_value = train_df['score'].max()



# In[19]:


# Normalize the column to the new range and replace the original column
train_df['score'] = ((train_df['score'] - original_min_value) * (new_max_range - new_min_range) / (original_max_value - original_min_value)) + new_min_range




# In[15]:


models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/clip-ViT-B-32-multilingual-v1"
]

model_names = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "clip-ViT-B-32-multilingual-v1"
]


# In[16]:


print(model_names[0])
model = SentenceTransformer(models[0])

similarity_scores = []
for index, row in train_df.iterrows():
    embeddings = model.encode([row['sentence1'], row['sentence2']], convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # Append similarity score to the list
    similarity_scores.append(cosine_similarity.item())

print(similarity_scores[0:5])


# In[21]:


train_df['model_1']= similarity_scores


# In[22]:


train_df.head()


# In[28]:


print(model_names[1])
model = SentenceTransformer(models[1])

similarity_scores = []
for index, row in train_df.iterrows():
    embeddings = model.encode([row['sentence1'], row['sentence2']], convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # Append similarity score to the list
    similarity_scores.append(cosine_similarity.item())

print(similarity_scores[0:5])
train_df['model_2']= similarity_scores


# In[30]:


print(model_names[2])
model = SentenceTransformer(models[2])

similarity_scores = []
for index, row in train_df.iterrows():
    embeddings = model.encode([row['sentence1'], row['sentence2']], convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # Append similarity score to the list
    similarity_scores.append(cosine_similarity.item())

print(similarity_scores[0:5])
train_df['model_3']= similarity_scores


# In[24]:


print(model_names[3])
model = SentenceTransformer(models[3])

similarity_scores = []
for index, row in train_df.iterrows():
    embeddings = model.encode([row['sentence1'], row['sentence2']], convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # Append similarity score to the list
    similarity_scores.append(cosine_similarity.item())

print(similarity_scores[0:5])


# In[31]:


train_df.head()


# In[33]:


from sklearn.metrics import mean_squared_error, precision_score, recall_score, accuracy_score
import numpy as np

def evaluate_models(actual_column, *predicted_columns, threshold=0.5):
   
    results = {}

    for predicted_column in predicted_columns:
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(train_df[actual_column], train_df[predicted_column]))

        # Convert the predicted values to binary based on the threshold
        binary_predictions = (train_df[predicted_column] >= threshold).astype(int)

        # Convert actual values to binary based on the threshold
        binary_actuals = (train_df[actual_column] >= threshold).astype(int)

        # Calculate precision, recall, and accuracy
        precision = precision_score(binary_actuals, binary_predictions)
        recall = recall_score(binary_actuals, binary_predictions)
        accuracy = accuracy_score(binary_actuals, binary_predictions)

        # Store results in the dictionary
        results[predicted_column] = {'RMSE': rmse, 'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}

    return results


# Assuming train_df is your DataFrame containing the data
evaluation_results = evaluate_models('score', 'model_1', 'model_2', 'model_3', 'model_4')

# Print the results
for model, metrics in evaluation_results.items():
    print(f"Metrics for {model}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()


# In[34]:


output_file = 'evaluation_results.csv'


df_results = pd.DataFrame.from_dict(evaluation_results, orient='index')


df_results.to_csv(output_file)


# In[35]:


print(f"Results have been saved to '{output_file}'.")


# In[36]:


pip install topsis-taanisha-10210323


# In[38]:


get_ipython().system('topsis evaluation_results.csv "1,1,1,1" "-,+,+,+" answer.csv')


# In[39]:


FinalAnswer = pd.read_csv('answer.csv')


# In[40]:


FinalAnswer


# In[41]:


import matplotlib.pyplot as plt
import pandas as pd


# In[42]:


model_names = FinalAnswer['Unnamed: 0']
topsis_scores = FinalAnswer['Topsis Score']

# Plotting the bar graph
plt.figure(figsize=(5,5))
plt.bar(model_names, topsis_scores, color='blue')
plt.xlabel('Model Name')
plt.ylabel('Topsis Score')
plt.title('Topsis Score evaluation for different text classification models (Done by Taanisha)')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

# Show the plot
plt.tight_layout()
plt.show()


# In[ ]:




