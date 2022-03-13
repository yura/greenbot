import telebot
import os
import redis
from telebot import types
import requests

keybstart = ['Войти', 'Зарегистрироваться']
keyback = ['Назад']
keybcategories = ['Пластик', 'Металл', 'Бумага', 'Стекло', 'Одежда', 'Батарейки',
                  'Лампочки', 'Техника', 'Крышечки', 'Шины', 'Пленка', 'Зубные щётки',
                  'Градусники', 'Биология', 'Микроволновка']

fsm = redis.Redis(db=2)
password = redis.Redis(db=3)
email = redis.Redis(db=4)
fsm1 = redis.Redis(db=5)
token = ''
bot = telebot.TeleBot(token)
r = requests.get('https://greenbot.yuriykotlyarov.com/categories.json')
r = r.json()
keyboard = list((r[i]['name'] for i in range(len(r))))
d = {}
for i in range(len(r)):
    d[r[i]['name']] = r[i]['url']


@bot.message_handler(func=lambda msg: True)
def w(msg):
    if msg.text == '/start':
        markup = types.ReplyKeyboardMarkup(row_width=3)
        markup.add(*keyboard)
        bot.send_message(msg.chat.id, 'Привет, выбери категорию мусора.', reply_markup=markup)
    elif msg.text == 'Назад':
        markup = types.ReplyKeyboardMarkup(row_width=3)
        markup.add(*keybcategories)
        bot.send_message(msg.chat.id, 'Выбери категорию мусора.', reply_markup=markup)
    else:
        url = d[msg.text]
        req = requests.get(url).json()



bot.polling()
