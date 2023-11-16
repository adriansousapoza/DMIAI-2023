import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


import nltk
import random
import re

# Download NLTK resources (if not downloaded)
nltk.download('punkt')



def find_txt_files(path):
    txt_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

# write a function to open a text file return the last line as a string
def read_human_files(paths):
    human_files = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            last_line = lines[-1]
            human_files.append(last_line)
    return human_files

def read_files(paths, line_start = 0):
    file_contents = []

    for file_path in paths:
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                # Ignore the first three lines and store the rest
                remaining_lines = ''.join(lines[line_start:])
                file_contents.append(remaining_lines)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    return file_contents


def split_text_into_strings(text_file_path, rvs_generator):
    with open(text_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

        # Split the text into articles using '####' as the delimiter
        articles = text.split('####')

        # Initialize list to store strings of sentences within each article
        all_strings = []

        for article in articles:
            # Split each article into sentences
            sentences = nltk.sent_tokenize(article.strip())

            while sentences:
                if len(sentences) <= 3:
                    if len(sentences) == 1:
                        pass
                    else:
                        # Add the string to the list of strings
                        selected_sentences = ' '.join(sentences[:])
                        all_strings.append(selected_sentences)
                    break

                # Choose a random number of sentences between min and max
                num_sentences = rvs_generator()
                # Ensure we don't exceed the length of available sentences
                num_sentences = min(num_sentences, len(sentences))

                # Join the selected number of sentences into a string
                selected_sentences = ' '.join(sentences[:num_sentences])

                # Add the string to the list of strings
                all_strings.append(selected_sentences)

                # Remove the selected sentences from the remaining list
                sentences = sentences[num_sentences:]

        return all_strings
    

def get_sentence_data(df, inplace = True):
    """
    For each row in the dataframe, find the number of sentences,
    and the average and std number of words per sentence.
    """

    # Make a copy of the dataframe
    df = df.copy()

    # Get the number of sentences
    df['num_sentences'] = df['text'].apply(lambda x: len(x.split('.')))

    # Get the average number of words per sentence
    df['avg_words_per_sentence'] = df['text'].apply(lambda x: np.mean([len(sentence.split(' ')) for sentence in x.split('.')]))

    # Get the std number of words per sentence
    df['std_words_per_sentence'] = df['text'].apply(lambda x: np.std([len(sentence.split(' ')) for sentence in x.split('.')]))

    # Get the average number of words per sentence
    df['avg_words_per_sentence'] = df['text'].apply(lambda x: np.mean([len(sentence.split(' ')) for sentence in x.split('.')]))

    # Get the std number of words per sentence
    df['std_words_per_sentence'] = df['text'].apply(lambda x: np.std([len(sentence.split(' ')) for sentence in x.split('.')]))

    if inplace:
        return df
    else:
        return df[['num_sentences', 'avg_words_per_sentence', 'std_words_per_sentence']]
    

def analyze_sentences(df_text):

    """
    For each row in the dataframe, find the number of sentences,
    the average and std number of words per sentence, the average and std number of words between sentences,
    and the avg. no. of abbreviations and capital letters per sentence.

    
    Parameters
    ----------
    df_text: pd.DataFrame
        Dataframe containing the text column

    Returns
    -------
    sentence_count, mean_words_per_sentence, std_words_per_sentence, mean_diff_words, std_diff_words, avg_abbrev_per_sentence, \
        avg_capital_letters_per_sentence, avg_dash_quest_semicolon_per_sentence, double_quest_excl_count

    """

    sentences = nltk.tokenize.sent_tokenize(df_text)
    sentence_count = len(sentences)


    if sentence_count == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    words_per_sentence = np.array([len(nltk.tokenize.word_tokenize(sentence)) for sentence in sentences])

    total_abbrev_count = 0
    total_capital_count = 0
    total_dash_quest_semicolon_count = 0
    total_double_quest_excl_count = 0
    total_excl_count = 0

    for sentence in sentences:
        # Count abbreviations using a simple regex pattern
        abbrev_count = len(re.findall(r'\b[A-Za-z]{2,}\.(?![a-z])', sentence))
        total_abbrev_count += abbrev_count
        
        # Count capital letters
        capital_count = sum(1 for c in sentence if c.isupper())
        total_capital_count += capital_count

        # Count dashes, question marks, and semicolons
        dash_quest_semicolon_count = sentence.count('-') + sentence.count('?') + sentence.count(';')
        total_dash_quest_semicolon_count += dash_quest_semicolon_count

        # Count double question marks and exclamation marks
        double_quest_excl_count = sentence.count('??') + sentence.count('!!') + sentence.count('?!') + sentence.count('!?')
        total_double_quest_excl_count += double_quest_excl_count

        # Count exclamation marks
        excl_count = sentence.count('!')
        total_excl_count += excl_count

    # Exclude ! from the count if it ends the last sentence or follows hej or spørgsmål
    last_sentence = sentences[-1]
    total_excl_count -= last_sentence.count('!')
    total_excl_count -= len(re.findall(r'\b(?:hej|spørgsmål)!', sentences[0]))




    if sentence_count > 1:
        # find mean and std for the length of consecutive sentences
        diff_words_between_sentences = np.diff(words_per_sentence)
        mean_diff_words = np.mean(diff_words_between_sentences)
        std_diff_words = np.std(diff_words_between_sentences)
    else:
        mean_diff_words = 0
        std_diff_words = 0
    

    avg_abbrev_per_sentence = total_abbrev_count / sentence_count
    avg_capital_letters_per_sentence = total_capital_count / sentence_count
    avg_dash_quest_semicolon_per_sentence = total_dash_quest_semicolon_count / sentence_count
    avg_excl_per_sentence = total_excl_count / sentence_count
 
    mean_words_per_sentence = np.mean(words_per_sentence)
    std_words_per_sentence = np.std(words_per_sentence)

    

    return sentence_count, mean_words_per_sentence, std_words_per_sentence, mean_diff_words, std_diff_words, avg_abbrev_per_sentence, \
        avg_capital_letters_per_sentence, avg_dash_quest_semicolon_per_sentence, total_double_quest_excl_count, avg_excl_per_sentence


def count_specific_words(text, word_lists):
    sentences = nltk.tokenize.sent_tokenize(text)
    sentence_count = len(sentences)
    
    # Initialize a list to store counts for each word list separately
    word_counts_per_list = [[] for _ in range(len(word_lists))]
    
    for sentence in sentences:
        for idx, word_list in enumerate(word_lists):
            # Initialize a counter for specified words in the current word list
            word_count = 0
            
            # Explicit variations of words in each word list to count
            for word in word_list:
                word_count += sentence.count(f'{word}')  # Space before and after
                word_count += sentence.count(f'{word.upper()}')
                # Add other variations as needed
            
            # Append the word count for the current word list
            word_counts_per_list[idx].append(word_count)
    
    # Calculate the average count per sentence for each word list
    avg_words_per_sentence = [sum(counts) / sentence_count if sentence_count > 0 else 0 for counts in word_counts_per_list]
    
    # return each value separately
    
    return avg_words_per_sentence

def preprocess(strings_df, save = False):

    # choose all columns e

    data = strings_df.copy()
    data['text'] = data['text'].astype('str')

    
    plural_pronouns = [' vi ', ' os ', ' vores ']

    human_fillers = [' jo ', ' jo,' ' jo!', ' lige ', ' sådan noget',  ' bestemt ',  ' bestemt!',  ' bestemt,',  ' bestemt.', ' gerne ', ' gerne!', ' gerne,', ' gerne.', ' rigtig god']
    ai_fillers = [' samt ', ' dette ', ' mens ', ' dog ']
    ai_fillers2 = [' en vis ', ' sammenfattende ']
    woke_list = [' tilbøjelig ', ' tilbøjelige ', ' parter ', ' grupper ', ' organisationer ', ' organisationer.', ' føle sig ', ' føler sig ']
    fake_news_list = [' subjektiv ', ' kilder ', ' kilder.', ' studier ', ' forklaringer ',
    ' manipulation ', ' disinformation ', ' disinformation.', ' misinformation ', ' misinformation.' ' da det ',]
    ai_words1 = [' vigtigt, at', ' vigtigt at ']
    ai_words2 = [' svært at ']

    word_lists = [human_fillers, ai_fillers, ai_fillers2, woke_list, fake_news_list, ai_words1, ai_words2]
    column_names = ['human_fillers', 'ai_fillers', 'ai_fillers2', 'woke', 'fake_news', 'ai_words1', 'ai_words2']



    data[['sentence_count', 'mean_words_per_sentence', 'std_words_per_sentence', 'neighbor_sentences_diff', \
      'neighbor_sentences_diff_std', 'abbrev_per_sentence', 'captials_per_sentence', 'avg_dash_quest_semicolon_per_sentence', 'double_quest_excl_count', 'avg_excl_per_sentence']] = data['text'].apply(analyze_sentences).apply(pd.Series)

    # Apply count_specific_words along with additional arguments using lambda function
    data[[col for col in column_names]] = data.apply(lambda row: count_specific_words(row['text'], word_lists), axis=1).apply(pd.Series)

    if save:
        data.to_csv('data_features.csv', index=False)

    # drop text column
    data = data.drop(['text'], axis=1).astype('float')
    return data


