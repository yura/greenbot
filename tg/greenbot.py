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


@bot.message_handler(func=lambda msg: True)
def w(msg):
    if msg.text == '/start':
        markup = types.ReplyKeyboardMarkup(row_width=3)
        markup.add(*keybcategories)
        bot.send_message(msg.chat.id, 'Привет, выбери категорию мусора.', reply_markup=markup)
    elif msg.text == 'Назад':
        markup = types.ReplyKeyboardMarkup(row_width=3)
        markup.add(*keybcategories)
        bot.send_message(msg.chat.id, 'Выбери категорию мусора.', reply_markup=markup)
    else:
        if msg.text == 'Батарейки':
            markup = types.ReplyKeyboardRemove()
            bot.send_message(msg.chat.id, 'Вот список мест:', reply_markup=markup)
            markup = types.ReplyKeyboardMarkup(row_width=2)
            markup.add(*keyback)
            bot.send_message(msg.chat.id, "ул. Ставропольская, 213", reply_markup=markup)

            bot.send_message(msg.chat.id, 'ул. Тургеневское шоссе, 27')
            bot.send_message(msg.chat.id, 'ул. Ставропольская, 222')
            bot.send_message(msg.chat.id, 'ул. Кубанская Набережная, 25')
            bot.send_message(msg.chat.id, 'пр-т Чекистов, 1/3')
            bot.send_message(msg.chat.id, 'ул. Сормовская ,108/1')
            bot.send_message(msg.chat.id, 'ул. им. 40-летия Победы, 144/5')
            bot.send_message(msg.chat.id, 'ул. Красная, 202')
            bot.send_message(msg.chat.id, 'ул. Красных Партизан, 173')
            bot.send_message(msg.chat.id, 'ул. Тургенева, 138/6')
            bot.send_message(msg.chat.id, 'ул. Московская, 54')
            bot.send_message(msg.chat.id, 'ул. Кореновская, 2, корпус 1')
            bot.send_message(msg.chat.id, 'ул. Петра Метальникова, 32/1')


bot.polling()
