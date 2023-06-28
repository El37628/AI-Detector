import re
import torch
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from transformers import RobertaModel, RobertaTokenizer
import nlpaug.augmenter.word as naw
from modAL.modAL.models import ActiveLearner
from modAL.modAL.uncertainty import uncertainty_sampling
import scipy
import scipy.sparse as sp
from random import shuffle
import discord
from discord.ext import commands
import asyncio

roberta_model_name = 'roberta-base'
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
roberta_model = RobertaModel.from_pretrained(roberta_model_name)

async def send_to_discord(ctx, msg):
    await ctx.send(msg)

def clean_text(text):
    """Cleans text by removing punctuation, stop words, and other noise."""
    text = contractions.fix(text)  # Expand contractions
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace

    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

def fit_vectorizers(reference_text):
    tf_vectorizer = TfidfVectorizer(stop_words='english')
    tf_vectorizer.fit([clean_text(reference_text)])

    count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))  # 1-gram to 3-gram
    count_vectorizer.fit([clean_text(reference_text)])

    return tf_vectorizer, count_vectorizer

def convert_text_to_bert_input(text, roberta_tokenizer, roberta_model):
    encoded_input = roberta_tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']
    with torch.no_grad():
        last_hidden_state = roberta_model(input_ids, attention_mask)[0]
    cls_embedding = last_hidden_state[:, 0, :].squeeze().numpy()  # Convert to numpy array
    return cls_embedding

async def calculate_augmented_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier):
    augmented_texts = augment_text(text)

    ai_probabilities = []
    for augmented_text in augmented_texts:
        transformed_text = convert_text_to_bert_input(augmented_text, roberta_tokenizer, roberta_model)

        ai_probability = classifier.predict_proba([transformed_text])[0][1]
        ai_probabilities.append(ai_probability)

    ai_percentage = np.mean(ai_probabilities) * 100
    human_percentage = 100 - ai_percentage

    print(f'Augmented Text - AI Writing Percentage: {ai_percentage:.2f}%')
    print(f'Augmented Text - Human Writing Percentage: {human_percentage:.2f}%')

    await send_to_discord(ctx, f'Augmented Text - AI Writing Percentage: {ai_percentage:.2f}%')
    await send_to_discord(ctx, f'Augmented Text - Human Writing Percentage: {human_percentage:.2f}%')

    return ai_percentage, human_percentage

async def calculate_tfidf_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier):
    augmented_texts = augment_text(text)

    X_pool = [convert_text_to_bert_input(t, roberta_tokenizer, roberta_model) for t in augmented_texts]
    y_pool = np.zeros(len(augmented_texts))

    ai_text = load_ai_text()
    human_text = load_human_text()

    X_train = [convert_text_to_bert_input(t, roberta_tokenizer, roberta_model) for t in ai_text + human_text]
    y_train = ['AI'] * len(ai_text) + ['Human'] * len(human_text)

    learner = ActiveLearner(
        estimator=classifier,
        X_training=X_train,
        y_training=y_train,
        query_strategy=uncertainty_sampling
    )

    num_iterations = 10
    for _ in range(num_iterations):
        if len(X_pool) == 0:
            break

        query_idx, query_inst = learner.query(X_pool, n_instances=1)
        if isinstance(query_idx, np.ndarray) and query_idx.size == 1:
            query_idx = query_idx.item()

        query_labels = get_true_labels(query_inst)
        learner.teach([X_pool[query_idx].tolist()], [query_labels[0]])

        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

    ai_percentage, human_percentage = await calculate_ai_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier)
    await send_to_discord(ctx, f'TF-IDF AI Writing Percentage: {ai_percentage:.2f}%')
    await send_to_discord(ctx, f'TF-IDF Human Writing Percentage: {human_percentage:.2f}%')

    return ai_percentage, human_percentage



async def calculate_ai_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier):
    cleaned_text = clean_text(text)
    text_embedding = convert_text_to_bert_input(cleaned_text, roberta_tokenizer, roberta_model)

    text_embedding_2d = text_embedding.reshape(1, -1)
    ai_probability = classifier.predict_proba(text_embedding_2d)[0][1]
    ai_percentage = ai_probability * 100
    human_percentage = 100 - ai_percentage

    return ai_percentage, human_percentage


def augment_text(text, aug_factor=5):
    augmented_texts = []
    aug = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=2)
    for _ in range(aug_factor):
        augmented_text = aug.augment(text)
        augmented_texts.append(augmented_text)
    return augmented_texts


def train_classifier(ai_text, human_text):
    ai_text_embeddings = [convert_text_to_bert_input(text, roberta_tokenizer, roberta_model) for text in ai_text]
    human_text_embeddings = [convert_text_to_bert_input(text, roberta_tokenizer, roberta_model) for text in human_text]

    ai_text_embeddings = [torch.tensor(embedding) for embedding in ai_text_embeddings]
    human_text_embeddings = [torch.tensor(embedding) for embedding in human_text_embeddings]

    X_train = torch.stack(ai_text_embeddings + human_text_embeddings)
    y_train = torch.tensor([1] * len(ai_text_embeddings) + [0] * len(human_text_embeddings))

    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)

    return roberta_tokenizer, roberta_model, classifier


def get_true_labels(query_instances):
    # Placeholder function to get true labels for queried instances
    # Replace with your actual labeling process
    if isinstance(query_instances, list):
        query_labels = [1] * len(query_instances)  # 1 for AI-generated, 0 for human-written
    else:
        query_labels = [1]  # 1 for AI-generated, 0 for human-written
    return query_labels


async def detect_ai_generated(ctx, text):
    ai_text = load_ai_text()
    human_text = load_human_text()

    roberta_tokenizer, roberta_model, classifier = train_classifier(ai_text, human_text)

    ai_percentage, human_percentage = await calculate_ai_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier)
    await send_to_discord(ctx, f'AI Writing Percentage: {ai_percentage:.2f}%')
    await send_to_discord(ctx, f'Human Writing Percentage: {human_percentage:.2f}%')

    await calculate_augmented_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier)
    await calculate_tfidf_percentage(ctx, text, roberta_tokenizer, roberta_model, classifier)
    return ai_percentage, human_percentage
 


def load_ai_text():
    """Load AI-generated text from a file."""
    with open('ai_text.txt', 'r', encoding='utf-8') as f:
        ai_text = f.readlines()

    # Shuffle the AI-generated text to ensure a diverse training dataset
    shuffle(ai_text)

    # Add more diverse AI-generated samples
    additional_ai_samples = [
        "The sun rose above the horizon, casting a warm glow over the sleepy town.",
        "She walked through the forest, marveling at the vibrant colors of the autumn leaves.",
        "The baby giggled and clapped her hands in delight as her mother played peek-a-boo.",
        # Add more samples here
    ]

    return [line.strip() for line in ai_text] + additional_ai_samples


def load_human_text():
    """Load human-generated text from a file."""
    with open('human_text.txt', 'r', encoding='utf-8') as f:
        human_text = f.readlines()

    # Shuffle the human-generated text to ensure a diverse training dataset
    shuffle(human_text)

    # Add more diverse human-written samples
    additional_human_samples = [
        "To be perfectly clear, I am not saying the Internet and technology will solve every human ill.",
        "The ability of science and technology to improve human life is known to us.",
        "The ability of science and technology to improve human life is known to us.",
        # Add more samples here
    ]

    return [line.strip() for line in human_text] + additional_human_samples


intents = discord.Intents.all()
intents.typing = False
intents.presences = False
intents.messages = True  # Enable the messages intent
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.command()
async def detect(ctx, *, text: str = None):
    if text is None:
        await ctx.send("Please provide some text with the command, like '!detect [text]'")
        return

    # Directly call the async function with await
    await detect_ai_generated(ctx, text)


@bot.event
async def on_ready():
    print(f'Bot is ready. Logged in as {bot.user}')

# Event to handle command errors
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return

if __name__ == '__main__':
    bot.run('MTEyMjc5MjY0Mjg1NDUzMTE1Mg.G6KiBq.HFwy_f0n4EmhEnb6T-AoCHERdEIz-hIyOfTl70')


