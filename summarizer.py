import os
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already done
nltk.download('stopwords')

# Function to parse the chat log
def parse_chat_log(file_path):
    with open(file_path, 'r') as file:
        chat_log = file.readlines()

    messages = {'user': [], 'ai': []}
    for line in chat_log:
        if line.startswith('User:'):
            messages['user'].append(line[6:].strip())
        elif line.startswith('AI:'):
            messages['ai'].append(line[4:].strip())
    
    return messages

# Function to get message statistics
def get_message_statistics(messages):
    total_messages = len(messages['user']) + len(messages['ai'])
    user_messages = len(messages['user'])
    ai_messages = len(messages['ai'])
    return total_messages, user_messages, ai_messages

# Function to clean and tokenize text
def clean_and_tokenize(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Remove non-word characters and lowercase
    tokens = text.split()
    return tokens

# Function to get top 5 most common keywords (without stop words)
def get_top_keywords(messages):
    all_messages = messages['user'] + messages['ai']
    all_text = ' '.join(all_messages)
    
    # Remove common stopwords
    stop_words = set(stopwords.words('english'))
    tokens = clean_and_tokenize(all_text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Count word frequencies
    word_counts = Counter(filtered_tokens)
    common_keywords = word_counts.most_common(5)
    
    return common_keywords

# Function to perform TF-IDF and get top keywords
def get_tfidf_keywords(messages):
    all_messages = messages['user'] + messages['ai']
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    tfidf_matrix = vectorizer.fit_transform(all_messages)
    tfidf_keywords = vectorizer.get_feature_names_out()
    
    return tfidf_keywords

# Function to generate summary
def generate_summary(messages):
    total_messages, user_messages, ai_messages = get_message_statistics(messages)
    
    # Get most common keywords
    common_keywords = get_top_keywords(messages)
    
    # Optional: Use TF-IDF for better keyword extraction
    tfidf_keywords = get_tfidf_keywords(messages)
    
    # Generate the summary
    summary = f"Summary:\n"
    summary += f"- The conversation had {total_messages} exchanges.\n"
    summary += f"- The user had {user_messages} messages, and the AI had {ai_messages} messages.\n"
    
    keyword_list = ', '.join([kw[0] for kw in common_keywords])
    summary += f"- Most common keywords (using word frequency): {keyword_list}.\n"
    
    tfidf_list = ', '.join(tfidf_keywords)
    summary += f"- Most common keywords (using TF-IDF): {tfidf_list}.\n"
    
    return summary

# Function to handle multiple chat logs in a folder
def summarize_multiple_chat_logs(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder {folder_path} does not exist.")
        return

    # Process each .txt file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            messages = parse_chat_log(file_path)
            summary = generate_summary(messages)
            print(f"Summary for {filename}:\n{summary}\n{'='*40}")
        else:
            print(f"Skipping non-text file: {filename}")

# Main function to run the summarization for a single chat log
def main(file_path):
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    messages = parse_chat_log(file_path)
    summary = generate_summary(messages)
    print(f"Summary for {file_path}:\n{summary}\n{'='*40}")

if __name__ == '__main__':
    # Single file processing
    # Replace with your chat log file path
    file_path = 'chat_example.txt'
    main(file_path)

    # Optional: Summarize all chat logs in a folder
    folder_path = './Chat_Logs'
    summarize_multiple_chat_logs(folder_path)
