import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# load texts and labels
def text_edit(text):
    # remove first two lines
    text = '\n'.join(text.split('\n')[2:])

    # cut longer than 2000 characters
    text = text[:2000]

    # # remove everything After Kilde:
    # text = text.split('Kilde:')[0]

    # remove empty lines
    # text = '\n'.join([line for line in text.split('\n') if line != ''])

    # remove new lines
    text = ' '.join(text.split('\n'))

    # remove double spaces
    # text = ' '.join(text.split('  '))

    # remove emojis
    # text = text.encode('ascii', 'ignore').decode('ascii')
   

    # Assuming 'text' is defined and contains sentences
    sentences = text.split('.')
    num_sentences = len(sentences)
    if num_sentences > 4:

        # Create an exponential distribution for probabilities
        # The exponential distribution should favor lower indices for rand_low and higher indices for rand_high
        probabilities = np.exp(np.linspace(0, 2, num_sentences-1))
        probabilities /= probabilities.sum()  # Normalize to make it a valid probability distribution

        # Choose rand_low and rand_high using the defined probabilities
        rand_low = np.random.choice(np.arange(num_sentences-1), p=probabilities[::-1])
        rand_high = np.random.choice(np.arange(rand_low+1, num_sentences), 
                        p=probabilities[rand_low:] / probabilities[rand_low:].sum())

        # Join the selected range of sentences
        text = '.'.join(sentences[rand_low:rand_high])

    return text

def load_data():

    import os

    data = {
        'human': [],
        'bot': []
    }

    data_sources = {
        'human' : ['data/heste-nettet-nyheder/'],
        'bot' : ['data/heste-nettet-nyheder-ai/gpt-3.5-turbo/', 'data/heste-nettet-nyheder-ai/gpt-4-0613/']
    }

    for source in data_sources:
        for path in data_sources[source]:
            for filename in os.listdir(path):
                with open(path + filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text = text_edit(content)
                    data[source].append(text)

    # cut to same length
    min_len = min(len(data['human']), len(data['bot']))
    data['human'] = data['human'][:min_len]
    data['bot'] = data['bot'][:min_len]
    
    # my_texts = np.array(data['human'] + data['bot'])
    # my_labels = np.array([0]*len(data['human']) + [1]*len(data['bot'])) 
    
        

    return data['human'], data['bot']

data_human, data_machine = load_data()

st.write(f'Length of human data: {len(data_human)}')
st.write(f'Length of machine data: {len(data_machine)}')


# create dataframe
df_human = pd.DataFrame(data_human, columns=['text'])
df_human['label'] = 'human'

df_machine = pd.DataFrame(data_machine, columns=['text'])
df_machine['label'] = 'machine'

df = pd.concat([df_human, df_machine], axis=0)

# get length of text (sentences, words, characters)
df['sentences'] = df['text'].apply(lambda x: len(x.split('.')))
df['words'] = df['text'].apply(lambda x: len(x.split(' ')))
df['characters'] = df['text'].apply(lambda x: len(x))


# define shared bins
bins_sentences = np.linspace(0, df['sentences'].max(), 50)
bins_words = np.linspace(0, df['words'].max(), 50)
bins_characters = np.linspace(0, df['characters'].max(), 50)

# get dist with np
dist_human_sentences = np.histogram(df[df['label']=='human']['sentences'], bins=bins_sentences)[0]
dist_machine_sentences = np.histogram(df[df['label']=='machine']['sentences'], bins=bins_sentences)[0]

dist_human_words = np.histogram(df[df['label']=='human']['words'], bins=bins_words)[0]
dist_machine_words = np.histogram(df[df['label']=='machine']['words'], bins=bins_words)[0]

dist_human_characters = np.histogram(df[df['label']=='human']['characters'], bins=bins_characters)[0]
dist_machine_characters = np.histogram(df[df['label']=='machine']['characters'], bins=bins_characters)[0]


# plot distribution
fig, ax = plt.subplots(1,3, figsize=(20,5))
ax[0].stairs(dist_human_sentences, bins_sentences, label='human')
ax[0].stairs(dist_machine_sentences, bins_sentences, label='machine')
ax[0].legend()
ax[0].set_title('Sentences')

ax[1].stairs(dist_human_words, bins_words, label='human')
ax[1].stairs(dist_machine_words, bins_words, label='machine')
ax[1].legend()
ax[1].set_title('Words')

ax[2].stairs(dist_human_characters, bins_characters, label='human')
ax[2].stairs(dist_machine_characters, bins_characters, label='machine')
ax[2].legend()
ax[2].set_title('Characters')
plt.close(fig)
st.pyplot(fig)





## Now lets look for common n-grams
# import nltk # install with: pip install nltk
# nltk.download('punkt')
# from nltk import ngrams


# def get_ngrams(text, n):
#     n_grams = ngrams(nltk.word_tokenize(text), n)
#     return [ ' '.join(grams) for grams in n_grams]

# more classical approach
def get_ngrams_simple(text, n):
    n_grams = set()
    for i in range(len(text)-n+1):
        n_grams.add(text[i:i+n])
    return n_grams

# df['ngrams'] = df['text'].apply(lambda x: get_ngrams(x, 2))
df['ngrams_simple'] = df['text'].apply(lambda x: get_ngrams_simple(x, 2))


# count ngrams
ngrams_human = {}
ngrams_machine = {}

for ngrams in df[df['label']=='human']['ngrams_simple']:
    for ngram in ngrams:
        if ngram not in ngrams_human:
            ngrams_human[ngram] = 0
        ngrams_human[ngram] += 1

for ngrams in df[df['label']=='machine']['ngrams_simple']:
    for ngram in ngrams:
        if ngram not in ngrams_machine:
            ngrams_machine[ngram] = 0
        ngrams_machine[ngram] += 1

# sort ngrams
ngrams_human = {k: v for k, v in sorted(ngrams_human.items(), key=lambda item: item[1], reverse=True)}

ngrams_machine = {k: v for k, v in sorted(ngrams_machine.items(), key=lambda item: item[1], reverse=True)}

# plot ngrams
fig, ax = plt.subplots(1,2, figsize=(20,5))
ax[0].bar(list(ngrams_human.keys())[:20], list(ngrams_human.values())[:20])
ax[0].set_title('Human n-grams')
ax[1].bar(list(ngrams_machine.keys())[:20], list(ngrams_machine.values())[:20])
ax[1].set_title('Machine n-grams')
plt.close(fig)
st.pyplot(fig)


## Look for symbols which occur more often in one of the two classes
# get symbols
symbols_human = {}
symbols_machine = {}

for text in df[df['label']=='human']['text']:
    for symbol in text:
        if symbol not in symbols_human:
            symbols_human[symbol] = 0
            symbols_machine[symbol] = 0
        symbols_human[symbol] += 1

for text in df[df['label']=='machine']['text']:
    for symbol in text:
        if symbol not in symbols_human:
            symbols_human[symbol] = 0
            symbols_machine[symbol] = 0
        symbols_machine[symbol] += 1

# sort symbols
symbols_human = {k: v for k, v in sorted(symbols_human.items(), key=lambda item: item[1], reverse=True)}

symbols_machine = {k: v for k, v in sorted(symbols_machine.items(), key=lambda item: item[1], reverse=True)}

# plot symbols
fig, ax = plt.subplots(1,2, figsize=(20,5))
ax[0].bar(list(symbols_human.keys())[:20], list(symbols_human.values())[:20])
ax[0].set_title('Human symbols')
ax[1].bar(list(symbols_machine.keys())[:20], list(symbols_machine.values())[:20])
ax[1].set_title('Machine symbols')
plt.close(fig)
st.pyplot(fig)


df