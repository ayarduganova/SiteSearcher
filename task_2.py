import os
import re
from bs4 import BeautifulSoup
import pymorphy3
from collections import defaultdict
import nltk
from nltk.corpus import stopwords


class TextProcessor:
    def __init__(self, pages_dir='/pages', index_file='index.txt'):
        self.pages_dir = pages_dir
        self.index_file = index_file

        # Загружаем стоп-слова из nltk
        self.download_nltk_resources()

        # Получаем русские стоп-слова из nltk
        self.stop_words = set(stopwords.words('russian'))

        # Добавляем дополнительные стоп-слова (при необходимости)
        extra_stops = {'это', 'который', 'мочь', 'весь', 'свой', 'твой', 'наш', 'ваш'}
        self.stop_words.update(extra_stops)

        # Инициализация pymorphy3
        self.morph = pymorphy3.MorphAnalyzer()

    def download_nltk_resources(self):
        """Загрузка необходимых ресурсов nltk"""
        try:
            # Проверяем наличие стоп-слов
            stopwords.words('russian')
        except LookupError:
            print("Загрузка стоп-слов из nltk...")
            nltk.download('stopwords')

    def extract_text_from_html(self, html_content):
        """Извлечение текста из HTML с помощью BeautifulSoup"""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Удаляем скрипты и стили
        for script in soup(['script', 'style', 'meta', 'link', 'noscript', 'select', 'option', 'button', 'input', 'textarea']):
            script.decompose()

        for element in soup.find_all(class_=re.compile(
                r'(translate|lang|menu|nav|footer|header|search|tab|breadcrumb|ad|banner|social|share)', re.I)):
            element.decompose()


        # Получаем текст
        text = soup.get_text()

        # Очищаем текст от лишних пробелов и переносов строк
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def tokenize(self, text):
        """Токенизация текста"""
        # Оставляем только слова из букв (русские и английские)
        tokens = re.findall(r'\b[а-яА-Яa-zA-Z]+\b', text.lower())

        # Фильтруем токены
        filtered_tokens = []
        for token in tokens:
            # Пропускаем стоп-слова (из nltk)
            if token in self.stop_words:
                continue

            # Пропускаем слишком короткие слова
            if len(token) <= 2:
                continue

            # Пропускаем числа и смешанные токены (уже отфильтрованы регуляркой)
            if token.isalpha():
                filtered_tokens.append(token)

        return filtered_tokens

    def lemmatize_tokens(self, tokens):
        """Лемматизация токенов и группировка по леммам"""
        lemma_groups = defaultdict(set)

        for token in tokens:
            # Получаем лемму для токена
            parsed = self.morph.parse(token)[0]
            lemma = parsed.normal_form

            # Добавляем токен в группу леммы
            lemma_groups[lemma].add(token)

        return lemma_groups

    def process_all_pages(self):
        """Обработка всех страниц из папки pages с сохранением результатов только для каждой страницы"""
        page_stats = []

        # Создаем папки для результатов по страницам
        tokens_dir = 'tokens_per_page'
        lemmas_dir = 'lemmas_per_page'
        os.makedirs(tokens_dir, exist_ok=True)
        os.makedirs(lemmas_dir, exist_ok=True)

        # Читаем индексный файл для получения URL
        url_by_page = {}
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                next(f)  # Пропускаем заголовок
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        page_num = int(parts[0])
                        url = parts[1]
                        url_by_page[page_num] = url

        # Обрабатываем каждый HTML файл
        for filename in sorted(os.listdir(self.pages_dir)):
            if filename.startswith('page_') and filename.endswith('.html'):
                filepath = os.path.join(self.pages_dir, filename)

                # Извлекаем номер страницы из имени файла
                page_num = int(filename.replace('page_', '').replace('.html', ''))
                url = url_by_page.get(page_num, 'Unknown URL')

                try:
                    # Читаем HTML файл
                    with open(filepath, 'r', encoding='utf-8') as f:
                        html_content = f.read()

                    # Извлекаем текст из HTML
                    text = self.extract_text_from_html(html_content)

                    # Токенизируем текст для текущей страницы
                    page_tokens = self.tokenize(text)

                    # Сохраняем токены для текущей страницы
                    self.save_page_tokens(page_num, page_tokens, url)

                    # Лемматизируем токены для текущей страницы
                    page_lemma_groups = self.lemmatize_tokens(page_tokens)

                    # Сохраняем леммы для текущей страницы
                    self.save_page_lemmas(page_num, page_lemma_groups, url)

                    # Сохраняем статистику по странице
                    page_stats.append((page_num, len(page_tokens), url))

                    print(f"✓ Страница {page_num}: {len(page_tokens)} токенов, {len(page_lemma_groups)} лемм")

                except Exception as e:
                    print(f"✗ Ошибка при обработке {filename}: {e}")

        # Выводим итоговую статистику
        print(f"\n{'=' * 50}")
        print(f"Обработано страниц: {len(page_stats)}")
        if page_stats:
            total_tokens = sum(stat[1] for stat in page_stats)
            print(f"Всего токенов (с повторениями): {total_tokens}")
            print(f"Среднее количество токенов на страницу: {total_tokens / len(page_stats):.1f}")

    def save_page_tokens(self, page_num, tokens, url):
        """Сохранение токенов для конкретной страницы"""
        tokens_filename = os.path.join('tokens_per_page', f'page_{page_num}_tokens.txt')

        with open(tokens_filename, 'w', encoding='utf-8') as f:
            f.write(f"Страница: {page_num}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Количество уникальных токенов: {len(tokens)}\n")
            f.write("-" * 50 + "\n\n")

            for token in sorted(tokens):
                f.write(f"{token}\n")

    def save_page_lemmas(self, page_num, lemma_groups, url):
        """Сохранение лемм для конкретной страницы"""
        lemmas_filename = os.path.join('lemmas_per_page', f'page_{page_num}_lemmas.txt')

        with open(lemmas_filename, 'w', encoding='utf-8') as f:
            f.write(f"Страница: {page_num}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Количество уникальных лемм: {len(lemma_groups)}\n")
            f.write("-" * 50 + "\n\n")

            for lemma, token_set in sorted(lemma_groups.items()):
                sorted_tokens = sorted(token_set)
                line = f"{lemma} >> " + " ".join(sorted_tokens)
                f.write(line + "\n")

    def run(self):
        """Основной метод для запуска обработки"""
        print("Начало обработки HTML файлов...")
        print(f"Используется {len(self.stop_words)} стоп-слов из библиотеки nltk")

        # Обрабатываем все страницы и сохраняем результаты только по отдельности
        self.process_all_pages()

        print(f"\nГотово! Результаты сохранены в папках:")
        print(f"- tokens_per_page/ (файлы с токенами для каждой страницы)")
        print(f"- lemmas_per_page/ (файлы с леммами для каждой страницы)")

    def run(self):
        """Основной метод для запуска обработки"""
        print("Начало обработки HTML файлов...")
        print(f"Используется {len(self.stop_words)} стоп-слов из библиотеки nltk")

        # Обрабатываем все страницы и сохраняем результаты только по отдельности
        self.process_all_pages()

        print(f"\nГотово! Результаты сохранены в папках:")
        print(f"- tokens_per_page/ (файлы с токенами для каждой страницы)")
        print(f"- lemmas_per_page/ (файлы с леммами для каждой страницы)")

if __name__ == "__main__":
    # Установка необходимых пакетов
    print("Проверка зависимостей...")
    required_packages = ['nltk', 'beautifulsoup4', 'pymorphy3']

    # Создаем и запускаем процессор
    processor = TextProcessor(pages_dir='pages', index_file='index.txt')
    processor.run()