import os
import re
import json
import math
from collections import defaultdict, Counter
from pymorphy3 import MorphAnalyzer
import numpy as np

class VectorSearch:
    def __init__(self,
                 lemmas_dir="lemmas_per_page",
                 index_file="index.txt",
                 tfidf_dir="tfidf_lemmas"):

        self.lemmas_dir = lemmas_dir
        self.index_file = index_file
        self.tfidf_dir = tfidf_dir

        self.morph = MorphAnalyzer()

        # doc_id -> url
        self.doc_urls = {}
        # список всех doc_id
        self.doc_ids = []
        # все леммы (словарь)
        self.vocabulary = []
        # lemma -> index в vocabulary
        self.lemma_to_idx = {}
        # doc_id -> tfidf вектор (numpy array)
        self.doc_vectors = {}
        # idf для всех лемм
        self.idf = {}
        # длина векторов для нормализации
        self.doc_norms = {}

    def load_doc_urls(self):
        """Загрузка соответствия doc_id -> URL"""
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Не найден {self.index_file}")

        with open(self.index_file, "r", encoding="utf-8") as f:
            f.readline()  # пропускаем заголовок
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    doc_id = int(parts[0])
                    url = parts[1]
                    self.doc_urls[doc_id] = url

    def load_tfidf_vectors(self):
        """Загрузка TF-IDF векторов из файлов"""
        if not os.path.exists(self.tfidf_dir):
            print(f"Папка {self.tfidf_dir} не найдена, создаём векторы...")
            self.build_tfidf_vectors()
            return

        # Собираем все уникальные леммы из всех файлов
        all_lemmas = set()
        tfidf_files = []

        for filename in os.listdir(self.tfidf_dir):
            if filename.endswith("_lemmas_tfidf.txt"):
                doc_id = int(filename.replace("page_", "").replace("_lemmas_tfidf.txt", ""))
                tfidf_files.append((doc_id, filename))
                self.doc_ids.append(doc_id)

        self.doc_ids.sort()
        tfidf_files.sort()

        # Первый проход: собираем все леммы
        for doc_id, filename in tfidf_files:
            path = os.path.join(self.tfidf_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        lemma = parts[0]
                        idf_val = float(parts[1])
                        all_lemmas.add(lemma)
                        self.idf[lemma] = idf_val

        # Создаём словарь
        self.vocabulary = sorted(all_lemmas)
        self.lemma_to_idx = {lemma: idx for idx, lemma in enumerate(self.vocabulary)}

        # Второй проход: строим векторы
        for doc_id, filename in tfidf_files:
            path = os.path.join(self.tfidf_dir, filename)
            vector = np.zeros(len(self.vocabulary))

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        lemma = parts[0]
                        tfidf_val = float(parts[2])
                        if lemma in self.lemma_to_idx:
                            vector[self.lemma_to_idx[lemma]] = tfidf_val

            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = np.linalg.norm(vector)

        print(f"Загружено {len(self.doc_ids)} документов, {len(self.vocabulary)} лемм")

    def build_tfidf_vectors(self):
        """Построение TF-IDF векторов из файлов лемм"""
        os.makedirs(self.tfidf_dir, exist_ok=True)

        # Считываем все леммы из всех документов
        doc_lemmas = {}  # doc_id -> Counter(lemma -> count)
        all_lemmas = set()

        for filename in os.listdir(self.lemmas_dir):
            if filename.endswith("_lemmas.txt"):
                doc_id = int(filename.replace("page_", "").replace("_lemmas.txt", ""))
                self.doc_ids.append(doc_id)

                path = os.path.join(self.lemmas_dir, filename)
                lemmas_counter = Counter()

                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith(("Страница:", "URL:", "Количество", "-")):
                            continue
                        if ">>" in line:
                            lemma = line.split(">>", 1)[0].strip()
                            lemmas_counter[lemma] += 1
                            all_lemmas.add(lemma)

                doc_lemmas[doc_id] = lemmas_counter

        self.doc_ids.sort()
        self.vocabulary = sorted(all_lemmas)
        self.lemma_to_idx = {lemma: idx for idx, lemma in enumerate(self.vocabulary)}

        # Считаем DF (document frequency)
        df = defaultdict(int)
        for doc_id, counter in doc_lemmas.items():
            for lemma in counter.keys():
                df[lemma] += 1

        # Считаем IDF
        N = len(self.doc_ids)
        for lemma in self.vocabulary:
            self.idf[lemma] = math.log(N / df[lemma]) if df[lemma] > 0 else 0.0

        # Строим TF-IDF векторы
        for doc_id in self.doc_ids:
            counter = doc_lemmas[doc_id]
            total = sum(counter.values())

            vector = np.zeros(len(self.vocabulary))
            tfidf_data = []

            for lemma, count in counter.items():
                tf = count / total if total > 0 else 0
                idf = self.idf.get(lemma, 0)
                tfidf = tf * idf

                if lemma in self.lemma_to_idx:
                    vector[self.lemma_to_idx[lemma]] = tfidf

                tfidf_data.append((lemma, idf, tfidf))

            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = np.linalg.norm(vector)

            # Сохраняем в файл
            out_path = os.path.join(self.tfidf_dir, f"page_{doc_id}_lemmas_tfidf.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for lemma in self.vocabulary:
                    idf_val = self.idf.get(lemma, 0)
                    tfidf_val = vector[self.lemma_to_idx[lemma]] if lemma in self.lemma_to_idx else 0
                    f.write(f"{lemma} {idf_val:.6f} {tfidf_val:.6f}\n")

        print(f"Построено {len(self.doc_ids)} TF-IDF векторов")

    def normalize_query_term(self, term: str) -> str:
        """Приведение слова к лемме"""
        term = term.lower()
        if not re.fullmatch(r"[а-яА-ЯёЁa-zA-Z]+", term):
            return ""
        return self.morph.parse(term)[0].normal_form

    def build_query_vector(self, query: str):
        """Построение TF-IDF вектора для запроса"""
        # Токенизация и лемматизация
        tokens = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]+\b', query.lower())
        lemmas = [self.normalize_query_term(t) for t in tokens]
        lemmas = [l for l in lemmas if l]

        if not lemmas:
            return np.zeros(len(self.vocabulary))

        # TF для запроса
        counter = Counter(lemmas)
        total = len(lemmas)

        # Строим вектор запроса
        query_vector = np.zeros(len(self.vocabulary))

        for lemma, count in counter.items():
            if lemma in self.lemma_to_idx:
                tf = count / total
                idf = self.idf.get(lemma, 0)
                query_vector[self.lemma_to_idx[lemma]] = tf * idf

        return query_vector

    def cosine_similarity(self, vec_a, norm_a, vec_b, norm_b):
        """Косинусное сходство между двумя векторами"""
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 10):
        """Векторный поиск по запросу"""
        query_vector = self.build_query_vector(query)
        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            return []

        # Считаем сходство с каждым документом
        scores = []
        for doc_id in self.doc_ids:
            doc_vector = self.doc_vectors[doc_id]
            doc_norm = self.doc_norms[doc_id]

            similarity = self.cosine_similarity(query_vector, query_norm, doc_vector, doc_norm)
            scores.append((doc_id, similarity))

        # Сортируем по убыванию сходства
        scores.sort(key=lambda x: x[1], reverse=True)

        # Возвращаем top_k результатов
        results = []
        for doc_id, score in scores[:top_k]:
            if score > 0:
                results.append({
                    "doc_id": doc_id,
                    "url": self.doc_urls.get(doc_id, "Unknown"),
                    "score": score
                })

        return results

    def search_with_snippet(self, query: str, top_k: int = 10):
        """Поиск с генерацией сниппетов"""
        results = self.search(query, top_k)

        for result in results:
            doc_id = result["doc_id"]
            snippet = self.get_snippet(doc_id, query)
            result["snippet"] = snippet

        return results

    def get_snippet(self, doc_id: int, query: str, max_length: int = 200):
        """Получение сниппета из документа"""
        # Читаем HTML файл
        html_path = os.path.join("pages", f"page_{doc_id}.html")
        if not os.path.exists(html_path):
            return ""

        try:
            from bs4 import BeautifulSoup
            with open(html_path, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Удаляем скрипты и стили
            for script in soup(['script', 'style', 'meta', 'link', 'noscript']):
                script.decompose()

            text = soup.get_text()
            # Очищаем от лишних пробелов
            text = ' '.join(text.split())

            # Ищем первое вхождение слова из запроса
            query_words = set(re.findall(r'\b[а-яА-ЯёЁa-zA-Z]+\b', query.lower()))

            # Находим позицию для сниппета
            start_pos = 0
            lower_text = text.lower()
            for word in query_words:
                pos = lower_text.find(word)
                if pos != -1:
                    start_pos = max(0, pos - 50)
                    break

            # Вырезаем сниппет
            snippet = text[start_pos:start_pos + max_length]
            if start_pos > 0:
                snippet = "..." + snippet
            if start_pos + max_length < len(text):
                snippet = snippet + "..."

            return snippet

        except Exception as e:
            return f"Ошибка при чтении: {e}"

    def build(self):
        """Полная инициализация"""
        self.load_doc_urls()
        self.load_tfidf_vectors()

    def cli(self):
        """Интерактивный CLI поиск"""
        print("\nВекторный поиск (TF-IDF)")
        print(f"Документов: {len(self.doc_ids)}, Лемм: {len(self.vocabulary)}")
        print("Пустая строка - выход\n")

        while True:
            try:
                query = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not query:
                break

            results = self.search_with_snippet(query)

            if not results:
                print("Ничего не найдено.\n")
                continue

            print(f"\nНайдено: {len(results)}")
            print("-" * 50)
            for i, res in enumerate(results, 1):
                print(f"{i}. [doc:{res['doc_id']}] score: {res['score']:.4f}")
                print(f"   {res['url']}")
                print(f"   {res['snippet'][:100]}...")
            print()


if __name__ == "__main__":
    engine = VectorSearch(
        lemmas_dir="lemmas_per_page",
        index_file="index.txt",
        tfidf_dir="tfidf_lemmas"
    )
    engine.build()
    engine.cli()
