import telebot
import os
import redis
from telebot import types
import requests

keybstart = ['Войти', 'Зарегистрироваться']
keybau = ['schedule', 'sign up', 'watch sign up', 'buy season ticket']
keybsch = ['last week', 'next week', 'back']
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



"""
@bot.message_handler(func=lambda msg: True)
def choosingfsmfunc(msg):
    print('hello there')
    if fsm.get(str(msg.chat.id)) == b'autorized':
        main(msg)
    else:
        start(msg)


def start(msg):
    print(fsm[str(msg.chat.id)])
    if 'start' in msg.text:
        markup = types.ReplyKeyboardMarkup(row_width=1)
        markup.add(*keybstart)
        bot.send_message(msg.chat.id, 'hello', reply_markup=markup)
    elif msg.text == 'Войти':
        markup = types.ReplyKeyboardRemove()
        bot.send_message(msg.chat.id, 'you need to share your phone number', reply_markup=markup)
        fsm[str(msg.chat.id)] = b'signin'
        markup = types.ReplyKeyboardMarkup()
        btn = types.KeyboardButton('Share your contact', request_contact=True)
        #btn = types.InlineKeyboardButton('Share your contact', request_contact=True)
        markup.add(btn)
        bot.send_message(msg.chat.id, "you need to press 'OK' to share this", reply_markup=markup)
    elif msg.text == 'Зарегистрироваться':
        fsm[str(msg.chat.id)] = 'singup'
        markup = types.ReplyKeyboardRemove()
        bot.send_message(msg.chat.id, 'input email address', reply_markup=markup)
    elif fsm[str(msg.chat.id)] == b'signin':
        print('here')
        bot.send_message(msg.chat.id, f(msg.text))
        fsm[str(msg.chat.id)] = 'password_singin'
    elif fsm[str(msg.chat.id)] == b'password_singin':
        if "b'" + msg.text + "'" == str(password[email[str(msg.chat.id)]]):
            fsm[str(msg.chat.id)] = 'autorized'
            fsm1[str(msg.chat.id)] = 'autorized'
            main(msg)
        else:
            bot.send_message(msg.chat.id, 'try again')
    elif msg.text == 'Зарегистрироваться':
        bot.send_message(msg.chat.id, 'input email')
        fsm[str(msg.chat.id)] = 'singup'
    elif '@' in msg.text and fsm[str(msg.chat.id)] == b'singup':
        email[str(msg.chat.id)] = msg.text
        bot.send_message(msg.chat.id, 'input password')
        fsm[str(msg.chat.id)] = 'password_signup'
    elif fsm[str(msg.chat.id)] == b'password_signup':
        password[email[str(msg.chat.id)]] = msg.text
        bot.send_message(msg.chat.id, 'hello nigger')
        fsm[str(msg.chat.id)] = 'autorized'
        fsm1[str(msg.chat.id)] = 'autorized'
        main(msg)
    else:
        bot.send_message(msg.chat.id, "don't know")


def main(msg):
    if fsm1[str(msg.chat.id)] == b'autorized':
        markup = types.ReplyKeyboardMarkup(row_width=2)
        markup.add(*keybau)
        bot.send_message(msg.chat.id, "you're autorized", reply_markup=markup)
        fsm1[str(msg.chat.id)] = 'zhak fresco'
    elif fsm1[str(msg.chat.id)] == b'back':
        markup = types.ReplyKeyboardMarkup(row_width=2)
        markup.add(*keybau)
        bot.send_message(msg.chat.id, "you're in main nigga", reply_markup=markup)
        fsm1[str(msg.chat.id)] = 'zhak fresco'
    elif msg.text == 'schedule':
        fsm1[str(msg.chat.id)] = 'schedule'
        markup = types.ReplyKeyboardRemove()
        bot.send_message(msg.chat.id, 'ok, schedule...', reply_markup=markup)
        markup = types.ReplyKeyboardMarkup(row_width=2)
        markup.add(*keybsch)
        bot.send_message(msg.chat.id, 'choose week nigga', reply_markup=markup)
    elif msg.text == 'sign up':
        print('here')
        bot.send_message(msg.chat.id, 'choose data')
        fsm1[str(msg.chat.id)] = 'choosing data'
    elif fsm1[str(msg.chat.id)] == b'choosing data':
        if True:
            bot.send_message(msg.chat.id, 'ok niger, dont forgot it')
        else:
            bot.send_message(msg.chat.id, 'no nigger you cant do it, choose another data')
    elif msg.text == 'watch sign up':
        bot.send_photo(msg.chat.id, 'https://static.wikia.nocookie.net/1a9327ff-6935-4b72-820c-f3ad6491e223')
    elif msg.text == 'buy season ticket':
        bot.send_message(msg.chat.id, '')
    elif msg.text == 'back':
        fsm1[str(msg.chat.id)] = b'back'
        main(msg)
    elif fsm1[str(msg.chat.id)] == b'schedule':
        bot.send_photo(msg.chat.id, 'https://www.meme-arsenal.com/memes/54c7ee322f4b0ae586ec96195a59a073.jpg')


@bot.message_handler(content_types=['contact'])
def contact(msg):
    print(phone.json())
    markup = types.ReplyKeyboardRemove()
    bot.send_message(msg.chat.id, 'your phone number recorded', reply_markup=markup)
"""
bot.polling()
