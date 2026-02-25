import requests
import time
import random
import os
from urllib.parse import urljoin, urlparse

class WebCrawler:
    
    def __init__(self, output_dir='pages', index_file='index.txt'):
        
        # папка для сохранения страниц
        self.output_dir = output_dir
        # файл с индексами
        self.index_file = index_file
        # cчетчик скачанных страниц
        self.downloaded_count = 0

        # Создаем папку для страниц если её нет
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Скачивание одной страницы
    def download_page(self, url, page_number):
        try:

            # Отправляем GET запрос
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Проверяем, что это HTML
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                print(f"Пропуск {url} - не HTML страница")
                return False

            # Сохраняем страницу с HTML разметкой
            filename = os.path.join(self.output_dir, f'page_{page_number}.html')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response.text)

            # Добавляем запись в индекс
            with open(self.index_file, 'a', encoding='utf-8') as f:
                f.write(f"{page_number}\t{url}\n")

            print(f"✓ Скачана страница {page_number}: {url}")
            return True

        except Exception as e:
            print(f"✗ Ошибка при скачивании {url}: {e}")
            return False

    # Скачивание страниц из списка
    # urls - список URL
    # max_pages - максимальное количество страниц
    # delay - задержка между запросами (в секундах)
    def crawl_from_list(self, urls, max_pages=100, delay=1):
        
        # Записываем файл индекса
        with open(self.index_file, 'w', encoding='utf-8') as f:
            f.write("Номер\tURL\n")

        # Начинаем нумерацию страниц с 1
        page_number = 1

        for url in urls:

            # Проверяем, не достигли ли лимита страниц
            if self.downloaded_count >= max_pages:
                break

            # Скачиваем страницу
            if self.download_page(url, page_number):
                self.downloaded_count += 1
                page_number += 1

                # Задержка для избежания блокировки
                sleep_time = random.uniform(delay, delay + 3)
                time.sleep(sleep_time)

        print(f"\n{'='*50}")
        print(f"Всего скачано страниц: {self.downloaded_count}")
        print(f"{'='*50}")



    
# Получение списка URL со статей https://dic.academic.ru/dic.nsf/enc_biology/
# start_id: начальный ID статьи
# end_id: конечный ID статьи
def get_academic_urls(start_id=9, end_id=112):
    urls = []
    base_url = "https://dic.academic.ru/dic.nsf/enc_biology/"

    for article_id in range(start_id, end_id + 1):
        url = f"{base_url}{article_id}"
        urls.append(url + "/")

    return urls



if __name__ == "__main__":
    print("Запуск краулера...")

    # Получаем список URL
    urls = get_academic_urls(1400, 1550)

    # Создаем краулер
    crawler = WebCrawler(output_dir='pages', index_file='index.txt')

    # Запускаем скачивание
    crawler.crawl_from_list(urls, max_pages=100, delay=0.5)

    print("\nГотово!")
