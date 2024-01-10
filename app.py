import streamlit as st
import numpy as np
# from saiga import model, tokenizer, generation_config, conversation, generate
from words_cloud import create_wordcloud_file
from file_handler import convert_to_txt


# streamlit run app.py

def generate_response(conversation):
    '''Генерация ответа на запрос к ЛЛМ'''
    # prompt = conversation.get_prompt(tokenizer)
    # with open('data/tmp/promt.txt', 'w+') as f:
    #     f.write(prompt)
    # output = generate(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt=prompt,
    #     generation_config=generation_config
    # )
    output = '''
    Основная тема: практическое применение искусственного интеллекта в различных сферах жизни и бизнеса.

    Главные тезисы:

    1) Развитие и продвижение 5 моделей (от 10 до 15).
    2) Показать рост хотя бы на 5% за золото.
    3) Веру в команду и мечту.
    4) Инвестирование в технологии и их практическое применение.
    5) Развитие и продвижение компании как крупнотехнологичной.
    6) Практическое применение нейросети и ценных инструментов для пользователей.
    7) Начало движения и поддержка от Сбера.
    8) Расширение предложения и развитие направлений.
    '''
    return output

# Конфиг страницы
st.set_page_config(
    page_title='🧙‍♂️✨ Команда "Графические нейромаги"',
    page_icon="🧙‍♂️",
    initial_sidebar_state="expanded",
)
st.title('Summary Generation📝')

# кнопка для ввода кастомного примера текста
def set_example_text(number):
    if number == '3':
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            with open(f'data/tmp/files/{uploaded_file.name}', 'wb') as f:
                f.write(bytes_data)
            text = convert_to_txt(f'data/tmp/files/{uploaded_file.name}', 'data/tmp/file_text.txt')
        else:
            text = ''
    else:
        with open(f'data/examples/demo_example_text{number}.txt', encoding='utf-8') as f:
            text = f.read()
    st.session_state.example_text = text

col1, col2, col3, *cols = st.columns(6)
with col1:
    btn_example_1 = st.button('Пример 1', on_click=set_example_text, args=['1'])
with col2:
    btn_example_2 = st.button('Пример 2', on_click=set_example_text, args=['2'])
with col3:
    btn_example_3 = st.button('Файл', on_click=set_example_text, args=['3'])

uploaded_file = st.file_uploader('Загрузка текстового PDF-документа')


# боковое меню
with st.sidebar:
    st.write("### ✅ Task:")
    st.info("Решение, позволяющее на базе текстового документа, полученного как протокол совещания/собрания, генерировать краткий отчёт.")

    st.write("### 🦌 LLM:")
    st.info("saiga_mistral_7b")

    st.divider()

    st.write("### 👨‍💻 Developed by:")
    st.info('Команда РТ Лабс 🧙‍♂️✨"Графические нейромаги"')

# Ввод текста
txt_input = st.text_area('Введите текстовую расшифровку собрания:', '',
                         height=275, key='example_text')

col1, col2, col3 = st.columns(3)
with col2:
    button_submit = st.button("Сгенерировать отчёт📄")

# при нажатии кнопки на генерацию
if button_submit:
    # подгружаем промпт для саммаризации отчёта
    with open('data/prompts/summarize.txt', encoding='utf-8') as f:
        sum_prompt = f.read()
    # добавляем его к тексту
    txt_input = sum_prompt + txt_input

    # сохраняем промпт+текст в историю
    with open('data/tmp/last_request.txt', 'w', encoding='utf-8') as f:
        f.write(txt_input)

    # conversation.add_user_message(txt_input)

    container = st.container()
    header = container.empty()
    header.write("Генерация отчёта...")
    placeholders = []
    for i in range(3):
        placeholder = container.empty()
        placeholders.append(placeholder)

    placeholders[0].status("Генерация текстового отчёта...")
    placeholders[1].status("Генерация облака слов...")
    placeholders[2].status("Генерация диаграммы...")

    # =============== Summary generation =============== 
    # генерация текстового саммари
    response = generate_response(conversation=None)
    # response = 'ОТВЕЕЕЕЕЕТТТТТТТТТ'
    create_wordcloud_file(txt_input)

    # сохраняем ответ модели в историю
    with open('data/tmp/last_response.txt', 'w', encoding='utf-8') as f:
        f.write(response)

    # =============== Diagram generation =============== 
    # подгружаем промпт для перевода в json-формат
    # with open('data/prompts/to_json.txt') as f:
    #     json_prompt = f.read()
    # conversation.add_user_message(json_prompt + response)

    # json_response = generate_response(conversation)

    # # сохраняем ответ модели в историю
    # with open('data/tmp/last_json.txt', 'w') as f:
    #     f.write(json_response)

    # =============== Отображение ответов =============== 
    placeholders[0].info(response)
    placeholders[1].image('data/wordscloud_data/wordcloud.png', caption='Облако слов совещания')
    with placeholders[2]:
        col1, col2, col3 = st.columns(3)
        with col2:
            with open('data/diagram/graph.drawio', encoding='utf-8') as f:
                st.download_button('Скачать mind-map по результатам совещания', f,
                                   file_name='mindmap.drawio')

    header.write("Генерация завершена!")
    # st.write(response)
