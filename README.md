# notbook_kaggle
خلاصه سازی متن یک خبر یا یک ویدیو یوتیوب 
# %% [markdown]
# # پروژه نهایی Capstone: ابزار خلاصه‌سازی خبر و ویدیو
# 
# در این نوتبوک از مدل‌های مولد خلاصه‌سازی ارائه‌شده توسط Hugging Face استفاده می‌کنیم. این ابزار به دو کاربرد اصلی می‌پردازد:
# 
# - **خلاصه‌سازی اخبار**: استفاده از دیتاست خبری (به عنوان نمونه از دیتاست cnn_dailymail)
# - **خلاصه‌سازی Transcript ویدیو YouTube**: استفاده از متنی نمونه (مثال transcript)
# 
# این نوتبوک علاوه بر نمایش نحوه بارگذاری مدل، خلاصه‌سازی متن، تنظیم پارامترها مانند `max_length` و `min_length`، همچنین نحوه استفاده از متغیرهای محیطی برای API Key (در صورت نیاز) را نشان می‌دهد.

# %% [markdown]
# ## نصب کتابخانه‌های مورد نیاز
# 
# اگر نیاز به نصب کتابخانه‌ها دارید، سلول زیر را اجرا کنید. (در اکثر نوتبوک‌های Kaggle این مرحله پیش‌فرض است)
# 
# ```python
# !pip install transformers datasets python-dotenv
# ```

# %% [code]
# اگر لازم بود کتابخانه‌ها رو نصب کنیم (برای محیط لوکال)؛ در Kaggle معمولاً این مراحل نیازی نیست.
# !pip install transformers datasets python-dotenv

# %% [markdown]
# ## تنظیمات اولیه و بارگذاری کتابخانه‌ها
# 
# در این سلول کتابخانه‌های لازم را وارد می‌کنیم و تنظیمات اولیه (مانند logging و بارگذاری متغیرهای محیطی) را انجام می‌دهیم.

# %% [code]
import os
import logging

# تلاش برای بارگذاری متغیرهای محیطی با استفاده از python-dotenv (برای لوکال)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.info("python-dotenv نصب نشده است؛ ادامه اجرا بدون آن.")

# وارد کردن کتابخانه‌های Hugging Face
from transformers import pipeline
from datasets import load_dataset

# تنظیم logging برای نمایش رویدادها
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("محیط اولیه تنظیم شد.")

# %% [markdown]
# ## تعریف توابع اصلی
# 
# - **load_summarizer**: بارگذاری مدل خلاصه‌سازی (پیش‌فرض: `facebook/bart-large-cnn`)  
# - **summarize_text**: خلاصه‌سازی متن ورودی با استفاده از مدل  
# - **fetch_sample_data**: دریافت داده نمونه؛ برای use_case = "news" از دیتاست cnn_dailymail و برای use_case = "youtube" از یک متن نمونه استفاده می‌کند.

# %% [code]
def load_summarizer(model_name="facebook/bart-large-cnn"):
    """
    بارگذاری مدل خلاصه‌سازی از Hugging Face.
    مدل پیش‌فرض: facebook/bart-large-cnn.
    """
    try:
        summarizer = pipeline("summarization", model=model_name)
        logging.info("✅ مدل خلاصه‌سازی بارگذاری شد.")
        return summarizer
    except Exception as e:
        logging.error("❌ خطا در بارگذاری مدل خلاصه‌سازی: %s", e)
        raise

def summarize_text(text, summarizer, max_length=130, min_length=30, do_sample=False):
    """
    خلاصه‌سازی متن با استفاده از مدل بارگذاری شده.
    """
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=do_sample)
        return summary[0]['summary_text']
    except Exception as e:
        logging.error("❌ خطا در خلاصه‌سازی متن: %s", e)
        raise

def fetch_sample_data(use_case):
    """
    دریافت داده نمونه بر اساس استفاده موردنظر.
    برای 'news': از دیتاست cnn_dailymail (نسخه 3.0.0) استفاده می‌شود.
    برای 'youtube': یک متن نمونه برای transcript ویدیو استفاده می‌شود.
    """
    if use_case == "news":
        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
            sample = dataset[0]["article"]
            logging.info("✅ نمونه خبر از دیتاست cnn_dailymail دریافت شد.")
            return sample
        except Exception as e:
            logging.error("❌ خطا در بارگذاری دیتاست خبری: %s", e)
            raise
    elif use_case == "youtube":
        sample = (
            "In this video, we explore the fundamentals of machine learning and deep learning. "
            "We break down the complex concepts into manageable pieces and demonstrate real-world applications "
            "of various ML algorithms and neural network architectures."
        )
        logging.info("✅ نمونه transcript ویدیو (YouTube) تنظیم شد.")
        return sample
    else:
        raise ValueError("Use case نامعتبر است. لطفاً 'news' یا 'youtube' را انتخاب کنید.")

# %% [markdown]
# ## اجرای خلاصه‌سازی در دو حالت: News و YouTube Transcript
# 
# در ادامه دو بلوک کد داریم:
# 
# 1. **خلاصه‌سازی خبر:** با استفاده از دیتاست cnn_dailymail.
# 2. **خلاصه‌سازی Transcript ویدیو YouTube:** با استفاده از یک متن نمونه.
# 
# شما می‌توانید با تغییر پارامترهای `max_length` و `min_length` تأثیر خروجی را مشاهده کنید.

# %% [code]
# استفاده از مدل خلاصه‌سازی
summarizer = load_summarizer()  # بارگذاری مدل

# -------------------------------
# حالت ۱: خلاصه‌سازی خبر
logging.info("شروع خلاصه‌سازی خبر...")
news_text = fetch_sample_data("news")
news_summary = summarize_text(news_text, summarizer, max_length=150, min_length=50)
print("=== خلاصه خبر ===")
print(news_summary)
print("\n")

# -------------------------------
# حالت ۲: خلاصه‌سازی Transcript ویدیو YouTube
logging.info("شروع خلاصه‌سازی transcript ویدیو YouTube...")
youtube_text = fetch_sample_data("youtube")
youtube_summary = summarize_text(youtube_text, summarizer, max_length=100, min_length=20)
print("=== خلاصه Transcript ویدیو YouTube ===")
print(youtube_summary)

# %% [markdown]
# ## جمع‌بندی
# 
# در این نوتبوک:
# - مدل `facebook/bart-large-cnn` از Hugging Face بارگذاری و دانلود شد.
# - دو نوع کاربرد خلاصه‌سازی بررسی شد: یکی برای خبر و دیگری برای transcript ویدیو.
# - می‌توانید با تغییر پارامترهای `max_length` و `min_length` خروجی خلاصه‌سازی را بهبود دهید.
# 
# این پروژه به عنوان یک نمونه از استفاده واقعی و کاربردی در دوره Generative AI می‌باشد و آماده ارایه به عنوان Capstone در Kaggle است.
# 
# برای بهبود بیشتر، می‌توانید:
# - ورودی‌های بیشتری اضافه کنید (مثلاً خواندن فایل‌های متنی واقعی)
# - خروجی‌ها را بصورت بصری (مثلاً استفاده از Streamlit) نمایش دهید.
# 


