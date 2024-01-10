from pdfminer.high_level import extract_text

def convert_to_txt(input_file, output_file):
    text = extract_text(input_file)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)
    return text