from flask import Flask, render_template, request
from task_5 import VectorSearch

app = Flask(__name__)

# инициализация поисковой системы
engine = VectorSearch(
    lemmas_dir="lemmas_per_page",
    index_file="index.txt",
    tfidf_dir="tfidf_lemmas"
)
engine.build()


@app.route("/", methods=["GET", "POST"])
def search():
    results = []
    query = ""

    if request.method == "POST":
        # получаем текст из поля ввода
        query = request.form.get("query", "")
        if query:
            # поиск и сохраняем 10 лучших результатов
            results = engine.search_with_snippet(query, top_k=10)

    # возвращаем HTML-шаблон
    return render_template("index.html", results=results, query=query)


if __name__ == "__main__":
    app.run(debug=True)