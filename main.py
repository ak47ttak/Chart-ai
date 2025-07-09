
import logging
import cv2
import numpy as np
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
from io import BytesIO
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

TOKEN = "AAF2bNIoYJ306RQSjGs-9vJregH_vnSTDhw"
logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("سلام! 📈 عکس چارتتو برام بفرست تا برات تحلیل کنم.")

def detect_trend(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, (20,100,100),(40,255,255))
    white = cv2.inRange(img, (200,200,200),(255,255,255))
    mask = cv2.bitwise_or(yellow, white)
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,80,minLineLength=40,maxLineGap=5)
    trend = "رنج"
    if lines is not None:
        slopes = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2!=x1: slopes.append((y2-y1)/(x2-x1))
        if slopes:
            m = np.mean(slopes)
            if m < -0.1: trend = "نزولی"
            elif m > 0.1: trend = "صعودی"
    return trend

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = await update.message.photo[-1].get_file()
    bio = BytesIO()
    await photo.download(out=bio)
    bio.seek(0)

    img = Image.open(bio).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    trend = detect_trend(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    analysis = f"🔎 روند کلی بازار: {trend}\n"
    if "strong buy" in text.lower(): analysis += "✅ سیگنال خرید قوی مشاهده شد.\n"
    if "strong sell" in text.lower(): analysis += "⚠️ سیگنال فروش قوی مشاهده شد.\n"
    if "RSI" in text: analysis += "✳️ احتمال سیگنال ورود از RSI.\n"
    if "MACD" in text: analysis += "✳️ تغییر قدرت روند از MACD.\n"

    forecast = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w, _ = forecast.shape
    color = (0,255,0) if trend == "صعودی" else (255,0,0)
    for i in range(15):
        s = (w - 100 + i * 4, int(h/2 - i * (1 if trend == "صعودی" else -1) * 2))
        e = (w - 96 + i * 4, int(h/2 - (i+1) * (1 if trend == "صعودی" else -1) * 2))
        cv2.arrowedLine(forecast, s, e, color, 1)

    isio = BytesIO()
    isio.name = "forecast.jpg"
    Image.fromarray(cv2.cvtColor(forecast, cv2.COLOR_BGR2RGB)).save(isio, format='JPEG')
    isio.seek(0)

    await update.message.reply_text(analysis)
    await update.message.reply_photo(photo=InputFile(isio))

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))
    app.run_polling()

if __name__ == "__main__":
    main()
