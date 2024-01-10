import re
import matplotlib.pyplot as plt

import natasha as nt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from stop_words import get_stop_words


# Установка пакетов для NLP
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('omw-1.4')


# Запись исходного и обработаного текста в файл
def write_text_in_file(text: str, raw_text=False):
    if raw_text:
        with open('data/tmp/Исходный текст.txt', 'w') as write_raw_text:
            write_raw_text.write(text)
    else:
        with open(f'data/tmp/Обработанный текст.txt', 'w') as write_processed_text:
            write_processed_text.write(text)


# Очищение текста от специальных символов, англ. слов,
# разделителей слов, сокращений и переносов
def clean_text(text: str):
    # Удаление переносов строк
    character_map = {
    ord('\n') : ' ',
    ord('\t') : ' ',
    ord('\r') : ' ',
    ord('ё')  : 'е'
    }
    text = text.translate(character_map)

    # Удаление специальных символов
    text = text.strip() # удаление пробелов в начале и конце строки
    text = re.sub(r"-", ' ', text) # разделение слов через дефис или слэш
    text = re.sub(r'[^\w\s]+|[\d]+', '', text) # удаление спец. символов
    text = re.sub(r'[^а-яА-ЯёЁ\d\s]', '', text) # удаление слов на английском языке, опционально
    text = re.sub(r'\b\w{,2}\b', '', text) # убираем слова из 2-х букв и меньше не из стоп-слов

    # Вызов функции обработки текста
    nlp_russian(text)


# Обработка русскоязычного текста 
def nlp_russian(text):
    # Инициализация объектов для преобразований и анализа текста
    morph_vocab = nt.MorphVocab()
    segmenter = nt.Segmenter()
    emb = nt.NewsEmbedding()
    morph_tagger = nt.NewsMorphTagger(emb)
    names_extractor = nt.NamesExtractor(morph_vocab)
    syntax_parser = nt.NewsSyntaxParser(emb)

    # Инициализация объекта из текста для преобразований
    doc = nt.Doc(text)

    # Токенизация и морфологический анализ токенов
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)

    stop_words = set(stopwords.words('russian'))

    # Лемматизация слов
    text_processed = ''
    for token in doc.tokens:
            if token.lemma not in stop_words:
                token.lemmatize(morph_vocab)
                text_processed = text_processed + " " + token.lemma

    # Запись обработанного текста в файл
    write_text_in_file(text_processed, raw_text=False)

    # Вызов функции генерации облака слов
    wordcloud_generation(text_processed)


# Настройка визуализации облака слов
def plot_cloud(wordcloud):
    plt.figure(figsize=(40, 30))
    plt.imshow(wordcloud) 
    plt.axis("off")


# Записываем в переменную стоп-слова русского языка
def wordcloud_generation(text: str):
    print('wordcloud_generation')
    print(f'{len(text)=}')
    STOPWORDS_RU = get_stop_words('russian')
    # Генерируем облако слов
    wordcloud = WordCloud(width = 2000, 
                      height = 1500, 
                      #random_state=1,
                      #background_color='black', 
                      margin=20, 
                      colormap='Blues', 
                      collocations=False, # учет коллокаций
                      stopwords = STOPWORDS_RU # доп. фильтрация стоп-слов
                      ).generate(text)
    # Визуализируем облако
    # plot_cloud(wordcloud)
    wordcloud.to_file('data/wordscloud_data/wordcloud.png')


def create_wordcloud_file(text: str=None):
    if text is None:
        with open('data/examples/example.txt', 'r') as f:
            text = f.read()
    print(len(text))
    text = clean_text(text)

# create_wordcloud_file()
