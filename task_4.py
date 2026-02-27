import os
import re
import math
from collections import Counter, defaultdict

class TFIDFCalculator:
    def __init__(self,
                 tokens_dir="tokens_per_page",
                 lemmas_dir="lemmas_per_page",
                 out_terms_dir="tfidf_terms",
                 out_lemmas_dir="tfidf_lemmas"):

        # папка с токенами
        self.tokens_dir = tokens_dir
        # папка с леммами
        self.lemmas_dir = lemmas_dir
        # папка для TF-IDF терминов
        self.out_terms_dir = out_terms_dir
        # папка для TF-IDF лемм
        self.out_lemmas_dir = out_lemmas_dir

        # список номеров документов
        self.doc_ids = []

        # списки терминов и лемм
        self.all_terms = []
        self.all_lemmas = []

        # tf терминов и лемм
        self.df_terms = defaultdict(int)
        self.df_lemmas = defaultdict(int)

        # idf терминов и лемм
        self.idf_terms = {}
        self.idf_lemmas = {}

    # вычисляем логарифм
    def _log(self, x):
        if x <= 0:
            return 0.0
        return math.log(x)

    # поиск документов page_X_tokens.txt и извлечение номеров
    def discover_doc_ids(self):
        if not os.path.isdir(self.tokens_dir):
            raise FileNotFoundError("Папка с токенами не найдена")

        doc_ids = []
        for filename in os.listdir(self.tokens_dir):

            # проверяем, соответствует ли имя шаблону
            match = re.match(r"page_(\d+)_tokens\.txt$", filename)
            if match:
                doc_id = int(match.group(1))
                doc_ids.append(doc_id)

        doc_ids.sort()
        return doc_ids

    # чтение файла и получение списка токенов для 1 файла
    def read_tokens_file(self, doc_id):
        path = os.path.join(self.tokens_dir, f"page_{doc_id}_tokens.txt")
        tokens = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # пропускаем служебные строки
                if (not line) or line.startswith(("Страница:", "URL:", "Количество", "-")):
                    continue

                # принимаем только слова из букв
                if re.fullmatch(r"[а-яА-ЯёЁa-zA-Z]+", line):
                    tokens.append(line.lower())
        return tokens

    # чтение лемм и формирование словаря для 1 файла
    def read_lemmas_mapping(self, doc_id):
        path = os.path.join(self.lemmas_dir, f"page_{doc_id}_lemmas.txt")

        lemma_map = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # пропускаем служебные строки
                if (not line) or line.startswith(("Страница:", "URL:", "Количество", "-")):
                    continue

                # строка должна содержать ">>"
                if ">>" not in line:
                    continue

                left, right = line.split(">>", 1)

                # убираем лишние пробелы и к нижнему регистру
                lemma = left.strip().lower()

                # разбиваем правую часть по пробелам,
                # убираем лишние пробелы и к нижнему регистру
                forms = [w.strip().lower() for w in right.split()]

                # каждой лемме записываем список форм
                lemma_map[lemma] = set(forms)

        return lemma_map

    # собираем общий словарь
    def build_global_vocabularies(self):
        terms_set = set()
        lemmas_set = set()

        # проходимся по всем документам и собираем все токены и леммы
        for doc_id in self.doc_ids:
            tokens = self.read_tokens_file(doc_id)
            terms_set.update(tokens)

            lemma_map = self.read_lemmas_mapping(doc_id)
            lemmas_set.update(lemma_map.keys())

        # сортируем
        self.all_terms = sorted(terms_set)
        self.all_lemmas = sorted(lemmas_set)

    # считаем число документов, в которых встретился термин/лемма хотя бы один раз
    def build_df(self):

        self.df_terms = defaultdict(int)
        self.df_lemmas = defaultdict(int)

        # Проходим по документам
        for doc_id in self.doc_ids:

            # читаем токены документа
            tokens = self.read_tokens_file(doc_id)

            # считаем, сколько раз каждый токен встретился
            token_counts = Counter(tokens)

            # берём множество уникальных токенов данного документа
            unique_terms_in_doc = set(tokens)

            # считаем df для токенов
            for term in unique_terms_in_doc:
                self.df_terms[term] += 1

            # читаем леммы документа
            lemma_map = self.read_lemmas_mapping(doc_id)

            # лемма считается встретившейся
            # если хотя бы одна её форма встретилась в токенах
            for lemma, forms in lemma_map.items():

                lemma_count = 0
                for form in forms:
                    lemma_count += token_counts.get(form, 0)

                if lemma_count > 0:
                    self.df_lemmas[lemma] += 1

    # считаем idf
    # IDF(term)  = log(N / df(term))
    # IDF(lemma) = log(N / df(lemma))
    # N — количество документов в коллекции
    # df(x) — число документов, где x встретился хотя бы 1 раз
    def build_idf(self):

        N = len(self.doc_ids)

        self.idf_terms = {}
        self.idf_lemmas = {}

        # IDF для всех терминов
        for term in self.all_terms:
            df = self.df_terms.get(term, 0)
            self.idf_terms[term] = self._log(N / df) if df > 0 else 0.0

        # IDF для всех лемм
        for lemma in self.all_lemmas:
            df = self.df_lemmas.get(lemma, 0)
            self.idf_lemmas[lemma] = self._log(N / df) if df > 0 else 0.0

    # подсчёт tf-idf и сохранение
    def compute_and_save(self):
        # создаём файлы
        os.makedirs(self.out_terms_dir, exist_ok=True)
        os.makedirs(self.out_lemmas_dir, exist_ok=True)

        for doc_id in self.doc_ids:

            # читаем токены документа
            tokens = self.read_tokens_file(doc_id)
            total_tokens = len(tokens)
            # ск-ко раз каждый токен встретился
            token_counts = Counter(tokens)

            # TF-IDF для терминов
            out_terms_path = os.path.join(self.out_terms_dir, f"page_{doc_id}_terms_tfidf.txt")

            with open(out_terms_path, "w", encoding="utf-8") as f:
                for term in self.all_terms:
                    # количество вхождений токена
                    tf = token_counts.get(term, 0)
                    # idf для этого токена
                    idf = self.idf_terms.get(term, 0.0)
                    # tfidf для этого токена
                    tfidf = tf * idf

                    f.write(f"{term} {idf:.6f} {tfidf:.6f}\n")

            # TF-IDF для лемм
            lemma_map = self.read_lemmas_mapping(doc_id)
            out_lemmas_path = os.path.join(self.out_lemmas_dir, f"page_{doc_id}_lemmas_tfidf.txt")

            with open(out_lemmas_path, "w", encoding="utf-8") as f:
                for lemma in self.all_lemmas:

                    # формы леммы
                    forms = lemma_map.get(lemma, set())

                    # сумма вхождений всех её форм
                    lemma_count = 0
                    for form in forms:
                        lemma_count += token_counts.get(form, 0)

                    # tf леммы: доля от общего числа токенов документа
                    tf = (lemma_count / total_tokens) if total_tokens > 0 else 0.0
                    # idf для леммы
                    idf = self.idf_lemmas.get(lemma, 0.0)
                    # tfidf для леммы
                    tfidf = tf * idf

                    f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")

            print(f"Документ {doc_id} обработан")

    def run(self):

        self.doc_ids = self.discover_doc_ids()
        print("Найдено документов:", len(self.doc_ids))

        self.build_global_vocabularies()
        self.build_df()
        self.build_idf()
        self.compute_and_save()

        print("Готово.")

if __name__ == "__main__":
    calculator = TFIDFCalculator()
    calculator.run()