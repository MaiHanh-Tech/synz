import logging
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import google.generativeai as genai

# --- CẤU HÌNH ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --- KẾT NỐI 2 BỘ NÃO (FLASH & PRO) ---
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_flash = genai.GenerativeModel('gemini-2.5-flash') # Chat thường
    model_pro = genai.GenerativeModel('gemini-2.5-pro')     # Chat sâu (/g)
else:
    print("⚠️ CẢNH BÁO: Chưa thấy GOOGLE_API_KEY!")

# Lưu lịch sử chat cho Flash
chat_history = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = """
    🚀 **VietMaiAI Lite (Siêu Tốc)** đã sẵn sàng!
    
    - Chat thường: Dùng Gemini Flash (Phản hồi tức thì).
    - Chat sâu: Gõ `/g <câu hỏi>` dùng Gemini Pro.
    
    *Phiên bản này đã bỏ Voice để đảm bảo tốc độ cao nhất.*
    """
    await update.message.reply_text(msg)

async def chat_with_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    chat_id = update.effective_chat.id
    
    print(f"📩 Nhận tin: {user_text}") 
    # Báo "Đang gõ..." ngay lập tức để Chị biết bot còn sống
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')

    try:
        ai_reply = ""

        # --- CHẾ ĐỘ CHUYÊN GIA (/g) ---
        if user_text.lower().startswith("/g "):
            real_prompt = user_text[3:].strip()
            # Dùng Pro, không cần nhớ lịch sử để tập trung phân tích
            response = model_pro.generate_content(real_prompt)
            ai_reply = f"🧠 **[PRO ANALYSIS]**\n{response.text}"

        # --- CHẾ ĐỘ THƯỜNG (FLASH) ---
        else:
            if chat_id not in chat_history:
                chat_history[chat_id] = model_flash.start_chat(history=[
                    {"role": "user", "parts": "Bạn là trợ lý ảo thông minh, trả lời ngắn gọn, súc tích và thân thiện."},
                    {"role": "model", "parts": "Dạ, em nghe đây ạ!"}
                ])
            chat = chat_history[chat_id]
            
            response = chat.send_message(user_text)
            ai_reply = response.text

        # --- GỬI KẾT QUẢ NGAY LẬP TỨC ---
        # Chia nhỏ nếu tin quá dài (Telegram giới hạn)
        if len(ai_reply) > 4000:
            for x in range(0, len(ai_reply), 4000):
                await update.message.reply_text(ai_reply[x:x+4000])
        else:
            await update.message.reply_text(ai_reply)
            
    except Exception as e:
        print(f"Lỗi: {e}")
        await update.message.reply_text(f"⚠️ Mạng chập chờn, chị hỏi lại giúp em nhé! ({e})")

# --- CHẠY BOT ---
if __name__ == '__main__':
    if not TELEGRAM_TOKEN:
        print("❌ LỖI: Chưa có TELEGRAM_TOKEN!")
    else:
        print("🚀 VietMaiAI Lite đang khởi động...")
        application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), chat_with_ai))
        application.run_polling()
