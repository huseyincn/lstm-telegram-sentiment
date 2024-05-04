# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import logging
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import numpy as np


# Load the model
new_model = tf.keras.models.load_model('savedmodeltech')


# Download NLTK resources (uncomment the following line if not already downloaded)?
nltk.download('punkt')
nltk.download('stopwords')


df = pd.read_csv('text.csv')
# Rename Columns
df.rename(columns={'text': 'Text', 'label': 'Label'}, inplace=True)
# Dropping the Index Colums
df.drop('Unnamed: 0',axis=1,inplace=True)
#METİNLERDEKİ GEREKSİZ KARAKTERLERİ TEMİZLİYORUZ
df['Text'] = df['Text'].str.replace(r'http\S+', '', regex=True)
df['Text'] = df['Text'].str.replace(r'[^\w\s]', '', regex=True)
df['Text'] = df['Text'].str.replace(r'\s+', ' ', regex=True)
df['Text'] = df['Text'].str.replace(r'\d+', '', regex=True)
df['Text'] = df['Text'].str.lower()
stop = stopwords.words('english')
df["Text"] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
X = df['Text']
y = df['Label']
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the text
#text="i dont know i feel so lost"

# # Preprocess the text
# text = text.lower()
# text = re.sub(r'http\S+', '', text)
# text = re.sub(r'[^\w\s]', '', text)
# text = re.sub(r'\s+', ' ', text)
# text = re.sub(r'\d+', '', text)
# text = ' '.join([word for word in text.split() if word not in stop])

# Define the tokenizer and fit it on the training data
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
maxlen = max(len(tokens) for tokens in X_train_sequences)


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def strTopadseq(message : str):
    message = message.lower()
    message = re.sub(r'http\S+', '', message)
    message = re.sub(r'[^\w\s]', '', message)
    message = re.sub(r'\s+', ' ', message)
    message = re.sub(r'\d+', '', message)
    message = ' '.join([word for word in message.split() if word not in stop])
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post')
    return padded_sequence

def emotionReturner(arr, n):
    emoText=""
    indices= np.argsort(arr.flatten())[-n:][::-1]
    emotions= str(indices)[1:-1].split(" ")
    for i in range(len(emotions)):
        tmp = emotions[i]
        if(tmp =='0'):
            emoText+="Sad - {:.5f} \n".format(arr[0][int(tmp)])
        elif(tmp=='1'):
            emoText+="Joy - {:.5f} \n".format(arr[0][int(tmp)])
        elif(tmp=='2'):
            emoText+="Love - {:.5f} \n".format(arr[0][int(tmp)])
        elif(tmp=='3'):
            emoText+="Anger - {:.5f} \n".format(arr[0][int(tmp)])
        elif(tmp=='4'):
            emoText+="Fear - {:.5f} \n".format(arr[0][int(tmp)])
        elif(tmp=='5'):
            emoText+="Suprise - {:.5f} \n".format(arr[0][int(tmp)])
    return emoText
    

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Help!")
# padded_sequence= strTopadseq(text)


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    padded_seq = strTopadseq(update.message.text)
    prediction = new_model.predict(padded_seq)
    result = emotionReturner(prediction,2)
    await update.message.reply_text(result)
# # Now you can use the `predict()` method
# predictions = new_model.predict(padded_sequence)


application = Application.builder().token("7132761340:AAG-m2kHzLQqnymo2NTFT3CdgK3PYSk36Jk").build()
# on different commands - answer in Telegram
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))
# on non command i.e message - echo the message on Telegram
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))
# Run the bot until the user presses Ctrl-C
application.run_polling(allowed_updates=Update.ALL_TYPES)
