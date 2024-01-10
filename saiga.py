import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


MODEL_NAME = "IlyaGusev/saiga_mistral_7b"
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."


class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_template=DEFAULT_RESPONSE_TEMPLATE
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{
            "role": "system",
            "content": system_prompt
        }]

    def add_user_message(self, message):
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def get_prompt(self, tokenizer):
        final_text = ""
        # for message in self.messages:
        #     message_text = self.message_template.format(**message)
        #     final_text += message_text
        message_text = self.message_template.format(**self.messages[-1])
        final_text = message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


def generate(model, tokenizer, prompt, generation_config):
    data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, max_length=1500, truncation=True)
    data = {k: v.to(model.device) for k, v in data.items()}
    output_ids = model.generate(
        **data,
        generation_config=generation_config,
    )[0]
    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    return output.strip()


config = PeftConfig.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    MODEL_NAME,
    torch_dtype=torch.float16
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.temperature = 0.01
generation_config.top_p = 0.9
generation_config.max_new_tokens = 2000
print(generation_config)

conversation = Conversation()

# inputs = ['''Выдели основные темы текста и напиши их по пунктам 1, 2, 3... и добавь в каждой теме короткое саммари по ней в двух предложениях: Звезды отражались в её глазах. Раньше, еще месяц назад их не было видно в черте города. Свет фонарей и смог заглушали их слабое мерцание. Все изменилось. И одним из немногих плюсов сложившейся ситуации было мерцание звезд в её глазах, и воздух, кажется стал чище. У всех у нас когда то была работа, и был дом. У некоторых были дети. У Лены была дочка. Она работала барменшой, а по вечерам подрабатывала в "клубе знакомств". Попросту говоря - была проституткой. Теперь ей уже не приходится ездить по незнакомым клиентам, каждый раз перед дверью квартиры креститься, и молиться, что бы все прошло как надо. Это тоже плюс. Но теперь у неё нет дочки. Она потерялась в первые дни, как только все это начиналось.
# Лена была "на вызове", когда исчезло электричество. Никто еще не знал, что это серьезно. Мобильная связь не работала, город погрузился во тьму за окнами однакомнатной квартиры, в которой возбужденный мужчина кончал в презерватив, а Лена считала секунды до очередного вызова. Она не могла как обычно принять душ, и вызвать такси, и после осознания этого, просто начала одеваться. Белье по привычке было сложено одной кучкой рядом с кроватью. Мужчина, имя которого она не захотела запоминать сказал ей спасибо и открыл дверь, что то проворчав напоследок на "долбанных электриков"...
# Лене было очень приятно выйти на свежий воздух, после пропахшей перегаром комнатушки. Она шла по темным улицам города, шла на "базу" пешком, и эта непроглядная тьма вокруг для неё сейчас была отражением внутреннего состояния, и поэтому она наслаждалась этой прогулкой. Она еще не знала, что электричество и водоснабжение уже не восстановят. Она не могла подумать, что через три часа её пятилетняя дочка, испугавшись темноты и одиночества, выйдет из квартиры, и пропадет навсегда. Она еще не знала, что её поиски будут бесполезны и опасны... Она просто шла по улице.''']

# with open('example.txt', 'r') as f:
#     inputs = [f.read()]

# prompt = '''
# ###Задание#### 
# Выдели основные темы текста приведенного ниже и добавь к каждой теме короткое саммари по каждой из них отдельно в двух предложениях.
# Ответ выдавай в следующем виде и никак иначе:
# 1) theme_1
# Summary: summary_1 for theme_1
# 2) theme_2
# Summary: summary_2 for theme_2
# 3) theme_3
# Summary: summary_3 for theme_3
# ...

# ###Текст###\n
# '''

# prompt = '''
# Сделай протокол из текста - который будет содержать тему встречи, главные и второстепенные вопросы, добавить к темам ответственных - если они есть, определить частотность упоминания темы и зафиксировать процентное соотношение по каждой теме:\n
# '''
# inputs[0] = prompt + inputs[0]

# with open('promt.txt', 'w+') as f:
#     f.write(inputs[0])

# for inp in inputs:
#     conversation = Conversation()
#     conversation.add_user_message(inp)
#     prompt = conversation.get_prompt(tokenizer)

#     output = generate(model, tokenizer, prompt, generation_config)
#     print(inp)
#     print("==============================")
#     print(output)
#     print()
#     print("==============================")
#     print()
