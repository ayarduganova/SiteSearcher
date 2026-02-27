import os
import re
import json
from collections import defaultdict
from pymorphy3 import MorphAnalyzer

class InvertedIndexSearch:
    def __init__(self, index_file="index.txt"):
        # Морфологический анализатор
        self.morph = MorphAnalyzer()
        # Файл с индексами
        self.index_file = index_file

        # lemma -> set(doc_id)
        self.inverted_index = defaultdict(set)
        # doc_id -> url
        self.doc_urls = {}
        # множество всех doc_id
        self.all_docs = set()

    # Чтение index.txt и запись в doc_urls (doc_id -> url)
    def load_doc_urls(self):
        self.doc_urls = {}

        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Не найден {self.index_file}")

        # Открываем файл для чтения
        with open(self.index_file, "r", encoding="utf-8") as f:

            # Пропускаем первую строку (заголовок)
            header = f.readline()

            # Читаем остальные строки файла
            for line in f:
                # Убираем пробелы по краям и разделяем строку по табуляции
                parts = line.strip().split("\t")

                # Первый элемент — номер документа
                doc_id = int(parts[0])

                # Второй элемент — URL страницы
                url = parts[1]

                # Добавляем запись в словарь
                self.doc_urls[doc_id] = url

    # Построение инвертированного индекса
    def build_index(self):
        self.load_doc_urls()

        # Очищаем индекс и множество документов перед построением
        self.inverted_index = defaultdict(set)
        self.all_docs = set()

        # Папка, где хранятся файлы с леммами
        lemmas_dir = "lemmas_per_page"

        # Получаем список всех файлов вида page_X_lemmas.txt
        lemma_files = sorted(
            fn for fn in os.listdir(lemmas_dir)
            if fn.startswith("page_") and fn.endswith("_lemmas.txt")
        )

        # Обрабатываем каждый файл
        for fn in lemma_files:

            # Из имени файла извлекаем номер документа
            doc_id = int(fn.replace("page_", "").replace("_lemmas.txt", ""))

            # Формируем полный путь к файлу
            path = os.path.join(lemmas_dir, fn)

            try:
                # Открываем файл
                with open(path, "r", encoding="utf-8") as f:

                    # Читаем файл построчно
                    for line in f:
                        line = line.strip()

                        # Пропускаем пустые строки и служебные заголовки
                        if not line or line.startswith(("Страница:", "URL:", "Количество", "-")):
                            continue

                        # Ожидаем строки формата:
                        # lemma >> token1 token2 ...
                        if ">>" not in line:
                            continue

                        # Берём левую часть до ">>" — это лемма
                        lemma_part = line.split(">>", 1)[0].strip()

                        # Добавляем номер документа в множество для этой леммы
                        self.inverted_index[lemma_part].add(doc_id)

                # Добавляем doc_id в множество всех документов
                self.all_docs.add(doc_id)

            except Exception as e:
                print(f"Ошибка на {fn}: {e}")

        print(f"Индекс построен: {len(self.inverted_index)} лемм, {len(self.all_docs)} документов")

    # Записываем инвертированный индекс в json файл
    def save_index(self, out_path="inverted_index.json"):
        # Преобразуем множества документов в списки
        data = {
            k: sorted(list(v)) for k, v in self.inverted_index.items()
        }

        # Записываем данные в файл
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Индекс сохранён в {out_path}")

    def load_index(self, path="inverted_index.json"):
        self.load_doc_urls()
        self.inverted_index = defaultdict(set)
        self.all_docs = set()

        # Открываем JSON-файл
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Проходим по всем леммам в файле
        for lemma, docs in data.items():

            # Преобразуем список документов обратно в множество
            self.inverted_index[lemma] = set(docs)

        # Формируем множество всех документов
        for docs in self.inverted_index.values():
            self.all_docs.update(docs)

        print(f"Индекс загружен: {len(self.inverted_index)} лемм, {len(self.all_docs)} документов")

    # Булев поиск

    # операторы
    OPERATORS = {"NOT", "AND", "OR"}
    # приоритеты
    PRECEDENCE = {"NOT": 3, "AND": 2, "OR": 1}
    # not
    NOT = {"NOT"}

    # Перевод слова запроса к лемме
    def normalize_query_term_to_lemma(self, term: str) -> str:
        term = term.lower()
        # Разрешаем только буквы (рус/англ) + ё
        if not re.fullmatch(r"[а-яА-ЯёЁa-zA-Z]+", term):
            return ""

        return self.morph.parse(term)[0].normal_form

    # Разбить строку запроса на токены
    def tokenize_query(self, query: str):
        # (, ), операторы, слова
        pattern = r"\(|\)|\bAND\b|\bOR\b|\bNOT\b|[а-яА-ЯёЁa-zA-Z]+"
        # Ищем все совпадения в строке
        raw = re.findall(pattern, query, flags=re.IGNORECASE)
        tokens = []

        # Обрабатываем каждый найденный элемент
        for t in raw:

            up = t.upper()
            # Если это логический оператор (AND, OR, NOT)
            if up in self.OPERATORS:
                tokens.append(up)

            # Если это скобка
            elif t in ("(", ")"):
                tokens.append(t)

            # Иначе это обычное слово
            else:
                # Приводим слово к лемме
                lemma = self.normalize_query_term_to_lemma(t)
                if lemma:
                    tokens.append(lemma)

        return tokens

    # Перевод выражения в постфиксную форму (RPN)
    # Используется алгоритм сортировочной станции (shunting-yard)
    def to_postfix(self, tokens):
        # Список для итоговой постфиксной записи
        output = []
        # Стек для хранения операторов и скобок
        stack = []

        # Проходим по каждому токену исходного выражения
        for tok in tokens:

            # Если токен — оператор
            if tok in self.OPERATORS:

                # Пока в стеке есть элементы
                while stack:
                    # Берём верхний элемент стека
                    top = stack[-1]

                    # Если верхний элемент — тоже оператор
                    if top in self.OPERATORS:

                        # Проверяем приоритет:
                        # если оператор в стеке имеет БОЛЬШИЙ приоритет
                        # или равный приоритет (и текущий не NOT),
                        # то выталкиваем его в выходной список
                        if (self.PRECEDENCE[top] > self.PRECEDENCE[tok]) or (
                                self.PRECEDENCE[top] == self.PRECEDENCE[tok] and tok not in self.NOT
                        ):
                            output.append(stack.pop())
                            continue

                    # Если приоритет меньше — прекращаем цикл
                    break

                # После обработки кладём текущий оператор в стек
                stack.append(tok)

            # Если токен — открывающая скобка
            elif tok == "(":
                # Просто кладём её в стек
                stack.append(tok)

            # Если токен — закрывающая скобка
            elif tok == ")":

                # Выталкиваем из стека всё в output,
                # пока не встретим открывающую скобку
                while stack and stack[-1] != "(":
                    output.append(stack.pop())

                # Если стек пуст, значит скобки несогласованы
                if not stack:
                    raise ValueError("Ошибка: лишняя ')' в запросе")

                # Удаляем открывающую скобку из стека
                stack.pop()

            else:
                # Иначе это термин (лемма)
                # Термины сразу добавляются в выходной список
                output.append(tok)

        # После обработки всех токенов
        # выгружаем оставшиеся элементы стека в output
        while stack:
            top = stack.pop()

            # Если остались скобки — ошибка
            if top in ("(", ")"):
                raise ValueError("Ошибка: несогласованные скобки")

            output.append(top)

        # Возвращаем постфиксную форму выражения
        return output

    # Вычисление постфиксного выражения (RPN)
    # На вход подаётся список токенов в постфиксной форме,
    def eval_postfix(self, postfix):
        # Стек для хранения промежуточных множеств документов
        st = []

        # Проходим по каждому токену постфиксного выражения
        for tok in postfix:

            # Если токен НЕ оператор, значит это лемма (термин поиска)
            if tok not in self.OPERATORS:
                # Получаем множество документов, где встречается эта лемма
                # Если леммы нет в индексе — возвращаем пустое множество
                st.append(set(self.inverted_index.get(tok, set())))
                continue

            # Если NOT
            if tok == "NOT":

                # Проверяем, есть ли операнд в стеке
                if not st:
                    raise ValueError("Ошибка: NOT без операнда")

                # Берём верхнее множество документов
                a = st.pop()

                # Выполняем инверсию:
                # все документы минус документы, содержащие слово
                st.append(self.all_docs - a)

            else:
                # Если оператор AND или OR (бинарные операторы)

                # Для бинарных операторов нужно минимум два операнда
                if len(st) < 2:
                    raise ValueError(f"Ошибка: оператор {tok} без двух операндов")

                # Снимаем два множества со стека
                b = st.pop()
                a = st.pop()

                # AND = пересечение множеств
                if tok == "AND":
                    st.append(a & b)

                # OR = объединение множеств
                elif tok == "OR":
                    st.append(a | b)

        # После вычисления в стеке должен остаться ровно один результат
        if len(st) != 1:
            raise ValueError("Ошибка: некорректный запрос (проверь операторы/скобки)")

        # Возвращаем итоговое множество документов
        return st[0]

    # Поиск
    def search(self, query: str):

        # Разбиваем строку запроса на токены
        tokens = self.tokenize_query(query)

        # Если запрос пустой — возвращаем пустой результат
        if not tokens:
            return set()

        # Переводим выражение в постфиксную форму
        postfix = self.to_postfix(tokens)

        # Вычисляем результат по постфиксной записи
        result_docs = self.eval_postfix(postfix)

        return result_docs

    # режим поиска
    def cli(self):

        print("\nБулев поиск")
        print("Операторы: AND OR NOT, скобки поддерживаются.\n")

        # Бесконечный цикл ввода запросов
        while True:
            # Считываем строку запроса и убираем пробелы по краям
            q = input("Запрос: ").strip()

            # Если строка пустая — завершаем программу
            if not q:
                break

            try:
                # Выполняем поиск
                docs = self.search(q)

                # Сортируем документы по номеру
                docs = sorted(docs)

                # Выводим количество найденных документов
                print(f"Найдено документов: {len(docs)}")

                # Показываем максимум первые 50 результатов
                for doc_id in docs[:50]:
                    # Получаем URL документа (если есть)
                    url = self.doc_urls.get(doc_id, "Unknown URL")

                    # Печатаем номер документа и ссылку
                    print(f"  {doc_id}\t{url}")

                if len(docs) > 50:
                    print("  ... (показаны первые 50 результатов)")

            except Exception as e:
                print(f"{e}")

# def run_queries_from_file(engine,
#                           query_file="query_examples.txt",
#                           output_file="query_results.txt"):
#
#     # Проверяем, существует ли файл с запросами
#     if not os.path.exists(query_file):
#         print(f"Файл {query_file} не найден")
#         return
#
#     with open(query_file, "r", encoding="utf-8") as f:
#         queries = [line.strip() for line in f if line.strip()]
#
#     with open(output_file, "w", encoding="utf-8") as out:
#
#         for query in queries:
#             out.write(f"Запрос: {query}\n")
#
#             try:
#                 # Выполняем поиск
#                 docs = sorted(engine.search(query))
#
#                 out.write(f"Найдено документов: {len(docs)}\n")
#
#                 # Пишем doc_id и URL
#                 for doc_id in docs:
#                     url = engine.doc_urls.get(doc_id, "")
#                     out.write(f"{doc_id}\t{url}\n")
#
#             except Exception as e:
#                 out.write(f"Ошибка: {e}\n")
#
#             out.write("\n" + "-"*50 + "\n\n")
#
#     print(f"✓ Результаты сохранены в {output_file}")

if __name__ == "__main__":
    engine = InvertedIndexSearch(index_file="index.txt")

    # строим индекс
    # engine.build_index()
    # engine.save_index("inverted_index.json")

    #load_index если уже строили
    engine.load_index("inverted_index.json")

    # булев поиск
    engine.cli()

    # run_queries_from_file(engine)