"""
MODES SPEC V5 — расширение V4 с фокусом на edge cases и pure-MCP паттерны.
==========================================================================

Изменения относительно V4:

1. РАСШИРЕНЫ ШАБЛОНЫ для проблемных actions из live agent session:
   - fs_search: добавлены формы с предлогами "in the project", "across",
     "по всему проекту", естественные глаголы (look for, search across)
   - git_commit: формы без target ("commit changes", "сделай коммит",
     "git commit", "сохрани изменения")
   - db_query: естественные формы ("query users table", "select from...",
     "выбери записи из", "получи данные из")
   - fs_write: формы без явного "create file" ("save to", "put in",
     "запиши результаты в")
   - все list actions: больше глаголов (show, display, what's in)

2. ДОБАВЛЕНЫ НОВЫЕ ACTIONS для покрытия популярных MCP-серверов:
   - http_get, http_post, http_put, http_delete (REST API)
   - schedule (cron / one-shot планирование)
   - notify (webhooks/notifications)
   - exec_shell (выполнение shell команды через MCP)
   - todo_create, todo_list, todo_complete (Linear/Notion-стиль)
   - email_send, email_search (Gmail-стиль)
   - vector_search, vector_upsert (Pinecone/Qdrant/Chroma)

3. БОЛЬШЕ ПАТТЕРНОВ caсual ("none") — дробные смыслы, которые провоцировали
   ложные срабатывания.

Итого: 50 actions = 41 из V4 + 9 новых.
"""

# Действия по доменам
ACTIONS_BY_DOMAIN = {
    "filesystem": [
        "fs_read", "fs_write", "fs_edit", "fs_delete",
        "fs_list", "fs_search", "fs_move", "fs_copy",
    ],
    "git": [
        "git_status", "git_commit", "git_push", "git_pull",
        "git_branch", "git_diff", "git_log",
    ],
    "github": [
        "gh_issue_create", "gh_issue_list", "gh_pr_create",
        "gh_pr_list", "gh_pr_merge", "gh_repo_search",
    ],
    "slack": [
        "slack_post", "slack_list_channels", "slack_search_msg",
    ],
    "db": [
        "db_query", "db_insert", "db_update", "db_delete",
        "db_list_tables", "db_describe",
    ],
    "web": [
        "web_search", "web_fetch",
    ],
    "memory": [
        "mem_store", "mem_recall", "mem_forget",
    ],
    "browser": [
        "browser_navigate", "browser_click", "browser_screenshot",
    ],
    "calendar": [
        "cal_create_event", "cal_list_events",
    ],
    # ─── NEW MCP-DOMAINS ───
    "http": [
        "http_get", "http_post", "http_put", "http_delete",
    ],
    "todo": [
        "todo_create", "todo_list", "todo_complete",
    ],
    "email": [
        "email_send", "email_search",
    ],
    "vector": [
        "vector_search", "vector_upsert",
    ],
    "exec": [
        "exec_shell",
    ],
    "control": [
        "none",
    ],
}

ACTIONS = [a for domain in ACTIONS_BY_DOMAIN.values() for a in domain]
# 41 (V4) + 13 (new) + 1 (none) = total — посчитаем динамически
N_ACTIONS = len(ACTIONS)

TARGETS = ["filename", "query", "folder", "url", "branch",
           "channel", "table", "issue_id", "pr_id", "event_id",
           "endpoint", "todo_id", "recipient", "vector_key", "command",
           "none"]
SCOPES = ["local", "global", "project", "system", "remote", "none"]
FORMATS = ["text", "json", "list", "raw", "markdown", "csv", "none"]
SPECIFICITIES = ["exact", "fuzzy", "broad", "none"]


# ===================================================================
# TEMPLATES — намеренно избыточные для проблемных actions
# ===================================================================

TEMPLATES = {
    # ─── filesystem ─────────────────────────────────────────────────
    "fs_read": {
        "ru": ["прочитай {t}", "открой {t}", "покажи содержимое {t}",
               "что в файле {t}", "выведи {t}", "посмотри {t}",
               "дай посмотреть {t}", "загрузи {t}"],
        "en": ["read {t}", "open {t}", "show contents of {t}",
               "what is in {t}", "print {t}", "display {t}",
               "let me see {t}", "load {t}", "cat {t}"],
        "target_kind": "filename",
    },
    "fs_write": {
        "ru": ["создай файл {t}", "запиши в {t}", "сохрани в {t}",
               "сохрани как {t}", "напиши файл {t}",
               "запиши результаты в {t}", "положи в {t}",
               "сделай файл {t}"],
        "en": ["create file {t}", "write to {t}", "save as {t}",
               "save to {t}", "put in {t}", "make file {t}",
               "create {t}", "new file {t}"],
        "target_kind": "filename",
    },
    "fs_edit": {
        "ru": ["отредактируй {t}", "измени {t}", "поправь {t}",
               "обнови {t}", "перепиши {t}", "правь {t}"],
        "en": ["edit {t}", "modify {t}", "update {t}",
               "rewrite {t}", "fix {t}", "change {t}"],
        "target_kind": "filename",
    },
    "fs_delete": {
        "ru": ["удали {t}", "сотри {t}", "убери {t}", "снеси {t}"],
        "en": ["delete {t}", "remove {t}", "rm {t}", "erase {t}"],
        "target_kind": "filename",
    },
    "fs_list": {
        "ru": ["покажи файлы в {t}", "что в папке {t}",
               "содержимое папки {t}", "список файлов в {t}",
               "ls {t}", "перечисли файлы в {t}"],
        "en": ["list files in {t}", "ls {t}", "show files in {t}",
               "what is in folder {t}", "directory listing of {t}",
               "show contents of folder {t}"],
        "target_kind": "folder",
    },
    "fs_search": {
        "ru": ["найди {t}", "найди {t} в проекте", "поищи {t}",
               "поищи {t} по всему проекту", "ищи {t}",
               "ищи {t} во всех файлах", "найди где {t}",
               "посмотри есть ли {t}", "грепни {t}"],
        "en": ["find {t}", "find {t} in the project",
               "find {t} across all files", "search for {t}",
               "search for {t} in the project", "look for {t}",
               "look up {t}", "grep {t}", "look for {t} everywhere",
               "where is {t}"],
        "target_kind": "query",
    },
    "fs_move": {
        "ru": ["перемести {t}", "перенеси {t}", "сдвинь {t}",
               "mv {t}"],
        "en": ["move {t}", "mv {t}", "relocate {t}"],
        "target_kind": "filename",
    },
    "fs_copy": {
        "ru": ["скопируй {t}", "сделай копию {t}", "cp {t}"],
        "en": ["copy {t}", "cp {t}", "duplicate {t}"],
        "target_kind": "filename",
    },

    # ─── git ────────────────────────────────────────────────────────
    "git_status": {
        "ru": ["git status", "статус репозитория", "что изменилось",
               "проверь статус", "покажи изменения"],
        "en": ["git status", "what changed", "show repo status",
               "check status", "any changes"],
        "target_kind": "none",
    },
    "git_commit": {
        "ru": ["закоммить {t}", "git commit {t}", "сделай коммит {t}",
               "коммить с сообщением {t}",
               # БЕЗ target — "commit changes"
               "закоммить изменения", "сделай коммит",
               "git commit", "commit", "сохрани изменения в гите",
               "коммит", "зафиксируй изменения"],
        "en": ["commit {t}", "git commit {t}", "make commit {t}",
               "commit with message {t}",
               "commit changes", "git commit", "commit",
               "make a commit", "save changes", "stage and commit"],
        "target_kind": "query",  # сообщение коммита
    },
    "git_push": {
        "ru": ["git push", "запушь", "пушни в репо", "отправь на сервер"],
        "en": ["git push", "push to remote", "push", "push changes"],
        "target_kind": "none",
    },
    "git_pull": {
        "ru": ["git pull", "подтяни изменения", "обнови локально",
               "пулл с сервера"],
        "en": ["git pull", "pull from remote", "pull", "fetch updates"],
        "target_kind": "none",
    },
    "git_branch": {
        "ru": ["переключись на {t}", "git checkout {t}",
               "новая ветка {t}", "создай ветку {t}", "branch {t}"],
        "en": ["switch to {t}", "git checkout {t}",
               "create branch {t}", "new branch {t}", "checkout {t}"],
        "target_kind": "branch",
    },
    "git_diff": {
        "ru": ["git diff", "покажи диф", "что в дифе",
               "какие изменения по строкам"],
        "en": ["git diff", "show diff", "show changes by line",
               "what is the diff"],
        "target_kind": "none",
    },
    "git_log": {
        "ru": ["git log", "история коммитов", "покажи коммиты",
               "последние коммиты"],
        "en": ["git log", "show commit history", "list commits",
               "recent commits"],
        "target_kind": "none",
    },

    # ─── github ─────────────────────────────────────────────────────
    "gh_issue_create": {
        "ru": ["создай issue {t}", "открой issue про {t}",
               "новый тикет {t}", "issue про {t} в гитхабе",
               "заведи баг {t}", "report bug {t}"],
        "en": ["create issue {t}", "open issue about {t}",
               "new issue {t}", "file an issue {t}",
               "create an issue about {t}", "report bug {t}",
               "open ticket {t}"],
        "target_kind": "query",
    },
    "gh_issue_list": {
        "ru": ["список issues", "покажи issues",
               "что открыто в гитхабе", "open issues"],
        "en": ["list issues", "show issues", "open issues",
               "what issues are open"],
        "target_kind": "none",
    },
    "gh_pr_create": {
        "ru": ["создай pull request {t}", "открой PR {t}",
               "новый PR про {t}"],
        "en": ["create pull request {t}", "open PR {t}",
               "new PR about {t}", "create PR {t}"],
        "target_kind": "query",
    },
    "gh_pr_list": {
        "ru": ["список PR", "покажи pull requests", "open PRs"],
        "en": ["list pull requests", "show PRs", "open PRs",
               "list open PRs"],
        "target_kind": "none",
    },
    "gh_pr_merge": {
        "ru": ["мерж PR {t}", "слей пулл реквест {t}",
               "merge PR {t}"],
        "en": ["merge PR {t}", "merge pull request {t}",
               "merge {t}"],
        "target_kind": "pr_id",
    },
    "gh_repo_search": {
        "ru": ["поищи репозитории {t}", "найди репо про {t}",
               "github search {t}"],
        "en": ["search repositories {t}", "find repo about {t}",
               "github search {t}", "search github for {t}"],
        "target_kind": "query",
    },

    # ─── slack ──────────────────────────────────────────────────────
    "slack_post": {
        "ru": ["напиши в слак {t}", "отправь сообщение в slack {t}",
               "запости в slack {t}", "slack post {t}",
               "post message to slack",
               "post to slack: {t}", "напиши в канал {t}"],
        "en": ["post to slack {t}", "send slack message {t}",
               "slack: {t}", "post message to slack",
               "post in slack: {t}", "send to channel {t}"],
        "target_kind": "query",
    },
    "slack_list_channels": {
        "ru": ["список каналов слак", "какие каналы в slack",
               "slack channels"],
        "en": ["list slack channels", "show channels",
               "what channels in slack"],
        "target_kind": "none",
    },
    "slack_search_msg": {
        "ru": ["поищи в slack {t}", "найди сообщения про {t} в slack",
               "search slack for {t}"],
        "en": ["search slack for {t}", "find messages in slack about {t}",
               "search messages {t} in slack"],
        "target_kind": "query",
    },

    # ─── db ─────────────────────────────────────────────────────────
    "db_query": {
        "ru": ["sql запрос {t}", "запрос к базе {t}",
               "выполни запрос {t}", "выбери записи из {t}",
               "получи данные из {t}", "select из {t}",
               "select from {t}", "запрос {t}",
               "получи строки из таблицы {t}",
               "выбери всё из {t}"],
        "en": ["query {t}", "query {t} table",
               "select from {t}", "select * from {t}",
               "run query on {t}", "get rows from {t}",
               "fetch from {t}", "get data from {t}",
               "select all from {t}", "list rows in {t}",
               "show rows from {t}"],
        "target_kind": "table",
    },
    "db_insert": {
        "ru": ["добавь запись в {t}", "insert в {t}",
               "вставь в таблицу {t}", "новая запись в {t}"],
        "en": ["insert into {t}", "add row to {t}",
               "insert row into {t}", "new record in {t}"],
        "target_kind": "table",
    },
    "db_update": {
        "ru": ["обнови записи в {t}", "update {t}",
               "поправь данные в {t}", "измени строки в {t}"],
        "en": ["update {t}", "update rows in {t}",
               "modify records in {t}"],
        "target_kind": "table",
    },
    "db_delete": {
        "ru": ["удали записи из {t}", "delete from {t}",
               "очисти таблицу {t}"],
        "en": ["delete from {t}", "delete rows from {t}",
               "clear table {t}", "truncate {t}"],
        "target_kind": "table",
    },
    "db_list_tables": {
        "ru": ["список таблиц", "какие таблицы в базе",
               "show tables"],
        "en": ["list tables", "show tables", "what tables exist",
               "list all tables"],
        "target_kind": "none",
    },
    "db_describe": {
        "ru": ["опиши таблицу {t}", "схема таблицы {t}",
               "describe {t}"],
        "en": ["describe table {t}", "schema of {t}",
               "describe {t}", "show schema {t}"],
        "target_kind": "table",
    },

    # ─── web ────────────────────────────────────────────────────────
    "web_search": {
        "ru": ["поищи в интернете {t}", "загугли {t}",
               "google {t}", "веб поиск {t}"],
        "en": ["search the web for {t}", "google {t}",
               "web search {t}", "google for {t}"],
        "target_kind": "query",
    },
    "web_fetch": {
        "ru": ["скачай страницу {t}", "запроси {t}",
               "fetch {t}", "загрузи URL {t}"],
        "en": ["fetch {t}", "download {t}", "get page {t}",
               "fetch URL {t}"],
        "target_kind": "url",
    },

    # ─── memory ─────────────────────────────────────────────────────
    "mem_store": {
        "ru": ["запомни {t}", "сохрани в память {t}",
               "memorize {t}"],
        "en": ["remember {t}", "store in memory {t}",
               "memorize {t}", "save to memory {t}"],
        "target_kind": "query",
    },
    "mem_recall": {
        "ru": ["вспомни {t}", "что я говорил про {t}",
               "достань из памяти {t}"],
        "en": ["recall {t}", "what did I say about {t}",
               "retrieve memory {t}"],
        "target_kind": "query",
    },
    "mem_forget": {
        "ru": ["забудь про {t}", "удали из памяти {t}"],
        "en": ["forget {t}", "forget about {t}",
               "remove from memory {t}"],
        "target_kind": "query",
    },

    # ─── browser ────────────────────────────────────────────────────
    "browser_navigate": {
        "ru": ["открой в браузере {t}", "перейди на {t}",
               "navigate to {t}"],
        "en": ["navigate to {t}", "open in browser {t}",
               "go to {t}", "browse {t}"],
        "target_kind": "url",
    },
    "browser_click": {
        "ru": ["кликни на {t}", "нажми {t}", "click {t}"],
        "en": ["click {t}", "click on {t}", "press {t}"],
        "target_kind": "query",
    },
    "browser_screenshot": {
        "ru": ["сделай скриншот", "скриншот страницы",
               "screenshot"],
        "en": ["take screenshot", "screenshot the page",
               "screenshot"],
        "target_kind": "none",
    },

    # ─── calendar ───────────────────────────────────────────────────
    "cal_create_event": {
        "ru": ["создай событие {t}", "встреча {t}",
               "запланируй {t}", "добавь в календарь {t}"],
        "en": ["create event {t}", "schedule meeting {t}",
               "add to calendar {t}", "new event {t}"],
        "target_kind": "query",
    },
    "cal_list_events": {
        "ru": ["что в календаре", "список событий",
               "мои встречи"],
        "en": ["list events", "show calendar",
               "what is on my calendar", "my meetings"],
        "target_kind": "none",
    },

    # ═══ NEW DOMAINS ════════════════════════════════════════════════

    # ─── http ───────────────────────────────────────────────────────
    "http_get": {
        "ru": ["GET {t}", "http get {t}", "запроси GET {t}",
               "запрос GET к {t}"],
        "en": ["GET {t}", "http get {t}", "make GET request to {t}",
               "send GET to {t}"],
        "target_kind": "endpoint",
    },
    "http_post": {
        "ru": ["POST {t}", "http post {t}", "отправь POST на {t}",
               "запрос POST к {t}"],
        "en": ["POST {t}", "http post {t}", "send POST to {t}",
               "post to {t}", "make POST request to {t}"],
        "target_kind": "endpoint",
    },
    "http_put": {
        "ru": ["PUT {t}", "http put {t}", "отправь PUT на {t}"],
        "en": ["PUT {t}", "http put {t}", "send PUT to {t}",
               "make PUT request to {t}"],
        "target_kind": "endpoint",
    },
    "http_delete": {
        "ru": ["DELETE {t}", "http delete {t}",
               "удали через DELETE {t}"],
        "en": ["DELETE {t}", "http delete {t}",
               "send DELETE to {t}"],
        "target_kind": "endpoint",
    },

    # ─── todo ───────────────────────────────────────────────────────
    "todo_create": {
        "ru": ["добавь задачу {t}", "новая todo {t}",
               "создай таску {t}", "добавь в список дел {t}"],
        "en": ["add todo {t}", "create task {t}",
               "new task {t}", "add to-do {t}"],
        "target_kind": "query",
    },
    "todo_list": {
        "ru": ["список задач", "todo list", "что в списке дел",
               "покажи задачи"],
        "en": ["list todos", "show tasks", "what is on my todo",
               "list tasks"],
        "target_kind": "none",
    },
    "todo_complete": {
        "ru": ["заверши задачу {t}", "отметь {t} выполненной",
               "complete {t}"],
        "en": ["complete task {t}", "mark {t} done",
               "finish todo {t}", "check off {t}"],
        "target_kind": "todo_id",
    },

    # ─── email ──────────────────────────────────────────────────────
    "email_send": {
        "ru": ["отправь письмо {t}", "напиши email {t}",
               "send email to {t}"],
        "en": ["send email to {t}", "email {t}",
               "compose email to {t}", "write email to {t}"],
        "target_kind": "recipient",
    },
    "email_search": {
        "ru": ["найди письма про {t}", "поищи в почте {t}",
               "search email for {t}"],
        "en": ["search email for {t}", "find emails about {t}",
               "look up emails {t}"],
        "target_kind": "query",
    },

    # ─── vector ─────────────────────────────────────────────────────
    "vector_search": {
        "ru": ["векторный поиск {t}", "найди похожее на {t}",
               "поиск в векторной базе {t}", "semantic search {t}"],
        "en": ["vector search {t}", "find similar to {t}",
               "semantic search {t}", "search embeddings for {t}"],
        "target_kind": "query",
    },
    "vector_upsert": {
        "ru": ["добавь в векторную базу {t}",
               "сохрани вектор {t}",
               "upsert вектор {t}"],
        "en": ["upsert vector {t}", "add embedding for {t}",
               "store in vector db {t}"],
        "target_kind": "vector_key",
    },

    # ─── exec ───────────────────────────────────────────────────────
    "exec_shell": {
        "ru": ["выполни команду {t}", "запусти shell {t}",
               "run shell {t}", "выполни в терминале {t}"],
        "en": ["run shell command {t}", "execute {t}",
               "run command {t}", "exec {t}",
               "run in terminal {t}"],
        "target_kind": "command",
    },
}


# ─── Модификаторы (как в V4) ───
SCOPE_MODIFIERS = {
    "local":   {"ru": ["в текущей папке", "локально", "тут", "здесь"],
                "en": ["in current folder", "locally", "here"]},
    "global":  {"ru": ["везде", "глобально", "по всей системе"],
                "en": ["everywhere", "globally", "system-wide"]},
    "project": {"ru": ["в проекте", "в репозитории", "по всему проекту"],
                "en": ["in the project", "in repo", "across the project",
                       "project-wide"]},
    "system":  {"ru": ["в системе", "на диске"],
                "en": ["in system", "on disk"]},
    "remote":  {"ru": ["удалённо", "на сервере", "в облаке"],
                "en": ["remotely", "on the server", "in cloud"]},
}

FORMAT_MODIFIERS = {
    "text":     {"ru": ["текстом", "как текст"], "en": ["as text", "in plain text"]},
    "json":     {"ru": ["в json", "в формате json"], "en": ["as json", "in json"]},
    "list":     {"ru": ["списком", "перечнем"], "en": ["as a list", "bullet points"]},
    "raw":      {"ru": ["сыро", "raw"], "en": ["raw", "as-is"]},
    "markdown": {"ru": ["в markdown"], "en": ["as markdown", "in md"]},
    "csv":      {"ru": ["в csv"], "en": ["as csv", "in csv format"]},
}

SPEC_MODIFIERS = {
    "exact": {"ru": ["точно", "именно", "конкретно"],
              "en": ["exactly", "precisely", "specifically"]},
    "fuzzy": {"ru": ["похожий", "что-то типа", "примерно"],
              "en": ["similar to", "something like", "roughly"]},
    "broad": {"ru": ["вообще", "в целом", "что-нибудь"],
              "en": ["anything", "in general", "any"]},
}


# ─── Расширенный пул casual фраз (none) ───
CASUAL = {
    "ru": [
        "что такое смысл жизни", "расскажи про квантовую физику",
        "как дела", "доброе утро", "спасибо", "пока",
        "интересная мысль", "согласен", "не знаю",
        "круто", "это сложно", "звучит хорошо",
        "хочу спать", "сегодня устал", "погода прекрасная",
        "что нового", "мне нравится осень", "люблю кофе",
        "правда?", "не уверен", "может быть", "возможно",
        "это интересный вопрос", "давай подумаем", "хорошо сказано",
        "ну ладно", "продолжай", "понял тебя",
    ],
    "en": [
        "what is the meaning of life", "tell me about physics",
        "good morning", "how are you", "thanks", "bye",
        "interesting thought", "I agree", "I don't know",
        "cool", "that is hard", "sounds good",
        "I am tired today", "weather is nice", "what is new",
        "I love coffee", "really?", "not sure", "maybe",
        "perhaps", "interesting question", "let me think",
        "well said", "alright", "go on", "got it",
        "I think so", "in my opinion", "honestly speaking",
        "you know what", "by the way", "anyway",
    ],
}


# ─── Целевые значения ───
FILE_NAMES = [
    "config.json", "readme.md", "main.py", "data.csv", "notes.txt",
    "report.pdf", "setup.cfg", "index.html", "style.css", "utils.py",
    "test.py", "draft.docx", "log.txt", "results.json", "model.pkl",
    "requirements.txt", "schema.sql", "app.js", "package.json", "todo.md",
    "Dockerfile", "Makefile", ".env",
]

QUERY_TERMS = [
    "TODO", "FIXME", "import numpy", "def main", "class Model",
    "error handling", "API key", "password", "deprecated", "version",
    "лицензия", "автор", "ошибка", "функция", "переменная",
    "конфигурация", "параметры", "результаты", "тесты", "logging",
    "login bug", "memory leak", "performance issue", "broken link",
    "security warning",
]

FOLDER_NAMES = [
    "src", "docs", "tests", "data", "config", "logs",
    "проект", "исходники", "build", "dist", "node_modules",
    "vendor", "examples", "scripts",
]

TABLE_NAMES = ["users", "orders", "products", "events",
               "logs", "audit", "messages", "sessions"]

URL_VALUES = ["https://example.com", "https://api.example.com/v1/data",
              "https://docs.python.org", "https://github.com/user/repo",
              "https://api.openai.com/v1/chat"]

ENDPOINT_VALUES = ["https://api.example.com/users",
                   "https://api.example.com/orders/123",
                   "/api/v1/items", "/api/auth/login",
                   "https://api.github.com/repos/owner/name/issues"]

BRANCH_NAMES = ["feature/login", "bugfix/cache", "main", "develop",
                "release/v2", "hotfix/auth", "experiment-x"]

PR_IDS = ["#42", "#100", "#7", "PR #15"]
ISSUE_IDS = ["#101", "#5", "issue #88"]
EVENT_IDS = ["event #1", "#meeting-42"]
TODO_IDS = ["task #1", "#5", "TODO-42"]

CHANNELS = ["#general", "#dev", "#random", "#alerts", "#frontend"]
RECIPIENTS = ["alice@example.com", "team@company.com",
              "bob.smith@org.io", "support@site.com"]
COMMANDS = ["ls -la", "ps aux", "df -h", "cat /etc/hosts",
            "docker ps", "pip install numpy"]
VECTOR_KEYS = ["doc-42", "embedding-user-7", "vector-id-123"]


def render_tags(action, target_kind, target_value, scope, fmt, spec):
    return (
        f"<action>{action}</action>"
        f"<target>{target_value}</target>"
        f"<scope>{scope}</scope>"
        f"<format>{fmt}</format>"
        f"<specificity>{spec}</specificity>"
    )


if __name__ == "__main__":
    print(f"Total actions: {N_ACTIONS} (V4 had 41)")
    print(f"By domain:")
    for d, acts in ACTIONS_BY_DOMAIN.items():
        print(f"  {d}: {len(acts)} ({', '.join(acts[:4])}{'...' if len(acts)>4 else ''})")
