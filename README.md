# AI Chat Log Summarizer

This project provides a Python tool for summarizing AI chat logs. It parses chat conversations, extracts statistics, and identifies key topics using both word frequency and TF-IDF analysis.

## Features

- Parses chat logs with "User:" and "AI:" message formats
- Calculates message statistics (total, user, AI)
- Extracts top keywords using word frequency (excluding stopwords)
- Extracts top keywords using TF-IDF
- Summarizes single files or all chat logs in a folder

## Requirements

- Python 3.6+
- scikit-learn
- nltk

Install dependencies:
```sh
pip install scikit-learn nltk
```

## Usage

### Summarize a Single Chat Log

Place your chat log (e.g., `chatExample.txt`) in the project directory. Run:

```sh
python summarizer.py
```

### Summarize All Chat Logs in a Folder

Place your chat logs in the `Chat_Logs/` folder. The script will automatically process all `.txt` files in that folder.

## File Structure

- [`summarizer.py`](summarizer.py): Main script for summarizing chat logs
- [`chatExample.txt`](chatExample.txt): Example chat log
- `Chat_Logs/`: Folder containing additional chat logs

## Example Output

```
Summary for chatExample.txt:
Summary:
- The conversation had 4 exchanges.
- The user had 2 messages, and the AI had 2 messages.
- Most common keywords (using word frequency): machine, learning, explain, field, ai.
- Most common keywords (using TF-IDF): machine, learning, explain, field, ai.
========================================
```

## Notes

- The script uses NLTK stopwords. The first run will download the stopwords corpus if not already present.
- Chat logs must use the format:
  ```
  User: message
  AI: message
  ```

## License

 [MIT LICENSE](./LICENSE)

---

**For junior Python developers interested in AI:**  
This project demonstrates basic NLP techniques for log summarization and is a good starting point for learning about text processing, keyword extraction, and Python scripting.
