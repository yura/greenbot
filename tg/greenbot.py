import telebot
import os
import redis
from telebot import types
import requests
import telegram_token

keybstart = ['Войти', 'Зарегистрироваться']
keyback = ['Назад']

categories = requests.get('https://greenbot.yuriykotlyarov.com/categories.json').json()

keyboard = []
category_urls = {}
for c in categories:
    keyboard.append(c['name'])
    category_urls[c['name']] = c['url']

fsm = redis.Redis(db=2)
password = redis.Redis(db=3)
email = redis.Redis(db=4)
fsm1 = redis.Redis(db=5)

token = telegram_token.TELEGRAM_TOKEN
bot = telebot.TeleBot(token)

def concat_addresses(category):
    addresses = []
    for r in category['recyclers']:
        addresses.append(r['address'])

    return "\n".join(addresses)

@bot.message_handler(func=lambda msg: True)
def w(msg):
    if msg.text == '/start':
        markup = types.ReplyKeyboardMarkup(row_width=3)
        markup.add(*keyboard)
        bot.send_message(msg.chat.id, 'Привет, выбери категорию вторсырья.', reply_markup=markup)
    elif msg.text == 'Назад':
        markup = types.ReplyKeyboardMarkup(row_width=3)
        markup.add(*keybcategories)
        bot.send_message(msg.chat.id, 'Выбери категорию вторсырья.', reply_markup=markup)
    else:
        url = category_urls[msg.text]
        category = requests.get(url).json()
        #bot.send_message(msg.chat.id, category['recyclers'][0]['address'])
        bot.send_message(msg.chat.id, concat_addresses(category))


bot.polling()
