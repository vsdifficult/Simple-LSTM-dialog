from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

# Conversations
conversations = [
    ("Hi", "Hi waht's up?"),
    ("What are your doing?", "Answer a questions"),
    ("What is your name?", "My name is ChatBot"),
    ("Bye", "Goodbay!"),
]

# Extracting questions and answers from conversations
questions, answers = zip(*conversations)

# Creating tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(questions) + list(answers))

# Preprocessing text to sequences
question_sequences = tokenizer.texts_to_sequences(questions)
answer_sequences = tokenizer.texts_to_sequences(answers)

vocab_size = len(tokenizer.word_index) + 1

# Padding sequences
max_sequence_len = max(
    [len(seq) for seq in question_sequences + answer_sequences])
padded_question_sequences = pad_sequences(question_sequences,
                                          maxlen=max_sequence_len,
                                          padding='pre')
padded_answer_sequences = pad_sequences(answer_sequences,
                                        maxlen=max_sequence_len,
                                        padding='pre')
padded_answer_sequences_one_hot = to_categorical(padded_answer_sequences,
                                                 num_classes=vocab_size)

# Building the model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_sequence_len))
model.add(LSTM(20, return_sequences=True))
model.add(Dense(vocab_size, activation='softmax'))

# Compiling the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Training the model
model.fit(padded_question_sequences,
          padded_answer_sequences,
          epochs=100,
          verbose=2)


# Function to generate response
def generate_response(question):
    question_seq = tokenizer.texts_to_sequences([question])
    question_seq = pad_sequences(question_seq,
                                 maxlen=max_sequence_len,
                                 padding='pre')
    predicted_index = np.argmax(model.predict(question_seq), axis=-1)
    predicted_word = tokenizer.sequences_to_texts(predicted_index.tolist())[0]
    return predicted_word


# Example usage
question = "Hi, how are you"
response = generate_response(question)
print(response)
