# =========================================================================
# PROGETTO: ANALISI DELLE TRAME E PREDIZIONE DEL VOTO DEI FILM
# =========================================================================

# ---- Librerie -----------------------------------------------------------
library(tidyverse)
library(tidytext)
library(ggwordcloud)
library(lubridate)
library(tm)
library(lda)
library(rsample)
library(glmnet)
library(randomForest)
library(caret)
library(Matrix)

# ---- Tema grafico globale -----------------------------------------------
colori_topic  <- c("#E63946","#457B9D","#2A9D8F","#E9C46A","#F4A261","#264653")
palette_bi    <- c("positivo" = "#2ecc71", "negativo" = "#e74c3c")
palette_class <- c("→ Alto (≥7)" = "#457B9D", "→ Basso (<7)" = "#E63946")

theme_movies <- theme_bw(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40", size = 11),
    axis.title    = element_text(size = 11),
    strip.text    = element_text(face = "bold"),
    strip.background = element_rect(fill = "gray92", color = NA),
    legend.position  = "bottom"
  )
theme_set(theme_movies)


# =========================================================================
# 1. CARICAMENTO E PULIZIA DEI DATI
# =========================================================================

movies <- read_csv("movies_dataset.csv", show_col_types = FALSE)

generi_lookup <- c(
  "28"="Action",    "12"="Adventure", "16"="Animation",
  "35"="Comedy",    "80"="Crime",     "99"="Documentary",
  "18"="Drama",     "10751"="Family", "14"="Fantasy",
  "36"="History",   "27"="Horror",    "10402"="Music",
  "9648"="Mystery", "10749"="Romance","878"="Sci-Fi",
  "53"="Thriller",  "10752"="War",    "37"="Western"
)

movies_clean <- movies %>%
  select(-backdrop_path, -poster_path, -original_title, -video, -adult) %>%
  filter(release_date <= "2025-12-31", vote_count >= 10) %>%
  mutate(
    year        = year(ymd(release_date)),
    genre_raw   = str_extract(genre_ids, "\\d+"),
    genre       = recode(genre_raw, !!!generi_lookup, .default = "Other"),
    high_rated  = factor(ifelse(vote_average >= 7, "Alto", "Basso"),
                         levels = c("Alto", "Basso")),
    n_parole    = str_count(overview, "\\S+"),
    n_caratteri = nchar(overview)
  ) %>%
  select(-genre_ids, -genre_raw, -release_date) %>%
  drop_na() %>%
  mutate(film_id = row_number())   # film_id aggiunto subito


# =========================================================================
# 2. PREPROCESSING DEL TESTO
# =========================================================================

text_preprocessing <- function(x) {
  x <- gsub("http\\S+\\s*", "", x)
  x <- gsub("#\\S+", "", x)
  x <- gsub("[[:cntrl:]]", "", x)
  x <- gsub("^[[:space:]]*|[[:space:]]*$", "", x)
  x <- gsub(" +", " ", x)
  x
}

movies_clean$overview <- gsub("\\b\\d{1,2}x\\d{2}\\b", "",
                              movies_clean$overview) %>%
  tolower() %>%
  removeWords(stopwords("SMART")) %>%
  text_preprocessing() %>%
  removePunctuation() %>%
  removeNumbers() %>%
  stripWhitespace() %>%
  removeWords(stopwords("SMART"))

data("stop_words")

parole_inutili <- tibble(word = c(
  "film", "movie", "story", "life", "one", "finds", "can", "will",
  "two", "new", "world", "_", "set", "takes", "make", "time", "back", 
  "love", 'family'
))


# =========================================================================
# 3. TOKENIZZAZIONE
# tidy_overview = dataframe lungo principale (contiene film_id e genre)
# =========================================================================

tidy_overview <- movies_clean %>%
  select(film_id, genre, overview) %>%
  unnest_tokens(output = word, input = overview) %>%
  anti_join(stop_words,     by = "word") %>%
  anti_join(parole_inutili, by = "word") %>%
  filter(nchar(word) > 2)           # rimuoviamo token troppo corti

word_counts <- tidy_overview %>% count(word, sort = TRUE)


# =========================================================================
# ANALISI ESPLORATIVA 
# =========================================================================
# --- 1. Distribuzione dei voti (Globale e divisa per classi <7 e >=7) ----
movies_clean %>%
  ggplot(aes(x = vote_average, fill = high_rated)) +
  geom_histogram(binwidth = 0.5, color = "white", alpha = 0.85) +
  scale_fill_manual(values = palette_class) +
  scale_x_continuous(breaks = seq(0, 10, by = 1)) +
  labs(
    title = "Distribuzione dei Voti (Vote Average)",
    subtitle = "Suddivisa in classi di voto: Alto (≥7) e Basso (<7)",
    x = "Voto Medio",
    y = "Numero di Film",
    fill = "Classe di Voto:"
  )

movies_clean %>%
  ggplot(aes(x = n_parole, fill = high_rated)) +
  geom_density(alpha = 0.6) +
  scale_fill_manual(
    values = c("Alto" = "#457B9D", "Basso" = "#E63946"),
    labels = c("Alto" = "→ Alto (≥7)", "Basso" = "→ Basso (<7)")
  ) +
  labs(
    title = "Distribuzione della Lunghezza delle Trame",
    subtitle = "Densità del numero di parole per film ad Alto e Basso voto",
    x = "Numero di Parole nella Trama",
    y = "Densità",
    fill = "Classe di Voto:"
  )

movies_clean %>%
  mutate(genre = fct_reorder(genre, vote_average, .fun = median, .desc = FALSE)) %>%
  ggplot(aes(x = genre, y = vote_average, fill = genre)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.7, outlier.alpha = 0.3) +
  geom_hline(yintercept = 7, linetype = "dashed", color = "#E63946", size = 0.8) +
  coord_flip() +
  labs(
    title = "Distribuzione dei Voti per Genere",
    subtitle = "La linea tratteggiata rossa indica la soglia del voto Alto (7)",
    x = "Genere",
    y = "Voto Medio"
  )
# Grafico a barre top 20 ------------------------------------------
word_counts %>%
  slice_max(n, n = 20) %>%
  mutate(word = fct_reorder(word, n)) %>%
  ggplot(aes(x = n, y = word, fill = n)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#a8d8ea", high = "#00487c") +
  scale_x_continuous(expand = expansion(mult = c(0, .05))) +
  labs(
    title    = "Le 20 parole più frequenti nelle trame",
    subtitle = paste0("Su ", nrow(movies_clean), " film"),
    x = "Frequenza", y = NULL
  )

# Wordcloud --------------------------------------------------------
word_counts %>%
  slice_max(order_by = n, n = 100) %>%
  ggplot(aes(label = word, size = n, color = n)) +
  geom_text_wordcloud(area_corr = TRUE, rm_outside = TRUE) +
  scale_size_area(max_size = 14) +
  scale_color_gradient(low = "gray60", high = "#c0392b") +
  theme_minimal() +
  labs(title = "Le parole più usate nelle trame dei film")


# =========================================================================
# 5. WORDCLOUD PER GENERE
# =========================================================================

generi_principali <- c("Action", "Drama", "Comedy", "Horror", "Thriller", "Romance")

word_counts_generi <- tidy_overview %>%
  filter(genre %in% generi_principali) %>%
  count(genre, word, sort = TRUE) %>%
  group_by(genre) %>%
  slice_max(order_by = n, n = 30, with_ties = FALSE) %>%
  ungroup()

word_counts_generi %>%
  ggplot(aes(label = word, size = n, color = genre)) +
  geom_text_wordcloud_area(shape = "circle", rm_outside = TRUE) +
  scale_size_area(max_size = 9) +
  facet_wrap(~ genre, ncol = 3) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text       = element_text(size = 12, face = "bold"),
    strip.background = element_rect(fill = "gray92", color = NA),
    plot.title       = element_text(face = "bold", size = 14)
  ) +
  labs(
    title    = "Wordcloud delle trame per genere",
    subtitle = "Le 30 parole più frequenti per ciascun genere"
  )


# =========================================================================
# 6. BIGRAMMI
# =========================================================================

bigrammi <- movies_clean %>%
  unnest_tokens(bigram, overview, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(
    !word1 %in% stop_words$word, !word2 %in% stop_words$word,
    !word1 %in% parole_inutili$word, !word2 %in% parole_inutili$word,
    !is.na(word1), !is.na(word2),
    nchar(word1) > 2, nchar(word2) > 2
  ) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE)

# Grafico a barre top 20
bigrammi %>%
  slice_max(order_by = n, n = 20, with_ties = FALSE) %>%
  mutate(bigram = fct_reorder(bigram, n)) %>%
  ggplot(aes(x = n, y = bigram, fill = n)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#f4a261", high = "#c0392b") +
  scale_x_continuous(expand = expansion(mult = c(0, .05))) +
  labs(
    title = "I 20 bigrammi più frequenti nelle sinossi",
    x = "Frequenza", y = NULL
  )

# Wordcloud bigrammi
bigrammi %>%
  slice_max(order_by = n, n = 80, with_ties = FALSE) %>%
  ggplot(aes(label = bigram, size = n, color = n)) +
  geom_text_wordcloud_area(shape = "square", rm_outside = TRUE) +
  scale_size_area(max_size = 8) +
  scale_color_gradient(low = "gray50", high = "#e76f51") +
  theme_minimal() +
  labs(title = "Wordcloud dei bigrammi più frequenti")


# =========================================================================
# 7. LUNGHEZZA TRAME PER GENERE
# =========================================================================

movies_clean %>%
  pivot_longer(cols = c(n_parole, n_caratteri),
               names_to = "metrica", values_to = "valore") %>%
  mutate(metrica = recode(metrica,
                          n_parole    = "Numero di parole",
                          n_caratteri = "Numero di caratteri")) %>%
  ggplot(aes(x = reorder(genre, valore, FUN = median),
             y = valore, fill = genre)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.2,
               outlier.size = 0.8, show.legend = FALSE) +
  facet_wrap(~ metrica, scales = "free_x") +
  coord_flip() +
  labs(
    title    = "Lunghezza delle sinossi per genere",
    subtitle = "Distribuzione del numero di parole e caratteri",
    x = NULL, y = NULL
  )


# =========================================================================
# 8. TF-IDF PER GENERE
# =========================================================================

tfidf_genre <- tidy_overview %>%
  filter(!is.na(genre), !is.na(word)) %>%
  count(genre, word, sort = TRUE) %>%
  bind_tf_idf(term = word, document = genre, n = n)

tfidf_genre %>%
  group_by(genre) %>%
  slice_max(tf_idf, n = 5, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(word = reorder_within(word, tf_idf, genre)) %>%
  ggplot(aes(tf_idf, word, fill = genre)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ genre, scales = "free_y", ncol = 4) +
  scale_y_reordered() +
  labs(
    title    = "Parole più distintive per genere (TF-IDF)",
    subtitle = "Le 5 parole con TF-IDF più alto in ciascun genere",
    x = "TF-IDF", y = NULL
  )


# =========================================================================
# 9. ANALISI LDA
# FIX principale: alcuni film dopo il filtraggio del vocabolario
# producono documenti vuoti → lda.collapsed.gibbs.sampler va in errore.
# Soluzione: filtrare i documenti vuoti PRIMA di passarli al sampler.
# =========================================================================

testi_per_film <- tidy_overview %>%
  group_by(film_id) %>%
  summarise(testo = paste(word, collapse = " "), .groups = "drop")

K <- 6
I <- 500

# Prima lexicalize completa per stimare frequenze vocabolario
corpus_lda_full <- lexicalize(testi_per_film$testo, lower = TRUE)
n_voc           <- word.counts(corpus_lda_full$documents, corpus_lda_full$vocab)

cat("Proporzione hapax (freq=1):", round(mean(n_voc == 1), 3), "\n")
cat("Proporzione freq<=5:       ", round(mean(n_voc <= 5), 3), "\n")

# Vocabolario ristretto: parole presenti in almeno 5 documenti
to.keep.voc <- corpus_lda_full$vocab[n_voc >= 5]

# Seconda lexicalize con vocabolario ristretto
corpus_lda <- lexicalize(testi_per_film$testo, vocab = to.keep.voc, lower = TRUE)

# FIX: rimuoviamo i documenti vuoti (NULL o matrice con 0 colonne)
# che emergono perché alcuni film hanno solo parole rare
doc_lengths  <- sapply(corpus_lda, function(d) {
  if (is.null(d) || !is.matrix(d)) return(0L)
  ncol(d)
})
keep_docs    <- which(doc_lengths > 0)

cat("Film rimossi per vocabolario vuoto:",
    nrow(testi_per_film) - length(keep_docs), "\n")
cat("Film usati per LDA:", length(keep_docs), "\n")

corpus_lda_clean   <- corpus_lda[keep_docs]
testi_per_film_lda <- testi_per_film[keep_docs, ]

set.seed(12345)
result <- lda.collapsed.gibbs.sampler(
  documents           = corpus_lda_clean,
  K                   = K,
  vocab               = to.keep.voc,
  num.iterations      = I,
  alpha               = 0.1,
  eta                 = 0.1,
  burnin              = 100,
  compute.log.likelihood = TRUE
)

# Convergenza
matplot(t(result$log.likelihoods), type = "l", col = "steelblue",
        xlab = "Iterazione", ylab = "Log-likelihood",
        main = "Convergenza Gibbs Sampler – LDA")
abline(v = 100, lty = 2, col = "red")
legend("bottomright", "burnin", lty = 2, col = "red", bty = "n")

# Top words per topic
top.words <- top.topic.words(result$topics, num.words = 8, by.score = TRUE)

# ---- Assegna nomi ai topic DOPO aver visto top.words ----
# (adatta i nomi se le parole cambiano con il tuo dataset)
topic.names <- c(
  "Sci-Fi & Avventura",
  "Romance & Scuola",
  "Crimine & Città",
  "Dramma Familiare",
  "Thriller & Mistero",
  "Guerra & Azione"
)
colnames(top.words) <- topic.names
print(top.words)

# Theta: prob topic x documento
theta <- t(result$document_sums)
theta <- theta / rowSums(theta)

testi_per_film_lda <- testi_per_film_lda %>%
  mutate(
    topic_dominante = apply(theta, 1, which.max),
    topic_label     = topic.names[topic_dominante]
  )

movies_lda <- movies_clean %>%
  inner_join(
    testi_per_film_lda %>% select(film_id, topic_dominante, topic_label),
    by = "film_id"
  )

cat("\nFilm per topic:\n")
print(movies_lda %>% count(topic_label, sort = TRUE))

# --- Grafico 9a: distribuzione topic per genere --------------------------
movies_lda %>%
  filter(genre %in% generi_principali) %>%
  count(genre, topic_label) %>%
  group_by(genre) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = prop, y = topic_label, fill = topic_label)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = setNames(colori_topic, topic.names)) +
  scale_x_continuous(labels = scales::percent_format()) +
  facet_wrap(~ genre, ncol = 3) +
  labs(
    title    = "Distribuzione dei topic per genere",
    subtitle = "Proporzione di film assegnati a ciascun topic",
    x = "Proporzione", y = NULL
  )

# --- Grafico 9b: voto medio per topic ------------------------------------
movies_lda %>%
  ggplot(aes(x = reorder(topic_label, vote_average, FUN = median),
             y = vote_average, fill = topic_label)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.75,
               outlier.alpha = 0.2, outlier.size = 0.8) +
  scale_fill_manual(values = setNames(colori_topic, topic.names)) +
  coord_flip() +
  labs(
    title    = "Voto medio per topic LDA",
    subtitle = "Distribuzione di vote_average per ciascun topic",
    x = NULL, y = "Vote Average (TMDB)"
  )

# --- Grafico 9c: popolarità vs voto medio --------------------------------
movies_lda %>%
  group_by(topic_label) %>%
  summarise(
    pop_media  = mean(popularity,   na.rm = TRUE),
    voto_medio = mean(vote_average, na.rm = TRUE),
    n_film     = n(), .groups = "drop"
  ) %>%
  ggplot(aes(x = pop_media, y = voto_medio,
             size = n_film, color = topic_label)) +
  geom_point(alpha = 0.85) +
  ggrepel::geom_text_repel(aes(label = topic_label),
                           size = 3.5, show.legend = FALSE,
                           max.overlaps = 20) +
  scale_size_continuous(range = c(5, 14), name = "N film") +
  scale_color_manual(values = setNames(colori_topic, topic.names),
                     guide  = "none") +
  labs(
    title    = "Popolarità vs Voto medio per topic",
    subtitle = "Dimensione del punto = numero di film nel topic",
    x = "Popolarità media", y = "Voto medio"
  )


# =========================================================================
# 10. DTM (tm) + CORRELAZIONI + REGRESSIONE PENALIZZATA
# =========================================================================

corpus_tm <- movies_clean$overview %>%
  VectorSource() %>%
  VCorpus() %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(content_transformer(function(x)
    iconv(x, to = "ASCII//TRANSLIT", sub = ""))) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removeWords, parole_inutili$word) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus_tm) %>%
  removeSparseTerms(0.99)

cat("\nDTM dimensioni (film × termini):", dim(dtm), "\n")

bigdata <- as.matrix(dtm) %>%
  as.data.frame() %>%
  mutate(
    vote_average = movies_clean$vote_average,
    high_rated   = movies_clean$high_rated,
    genre        = movies_clean$genre,
    film_id      = movies_clean$film_id
  )

# --- 10a. Correlazioni parola-voto ---------------------------------------
correlazioni <- dtm %>%
  as.matrix() %>%
  as.data.frame() %>%
  mutate(vote_average = movies_clean$vote_average) %>%
  summarise(across(-vote_average,
                   ~ cor(.x, vote_average, use = "complete.obs"))) %>%
  pivot_longer(everything(),
               names_to  = "word",
               values_to = "correlazione") %>%
  arrange(desc(abs(correlazione)))

correlazioni %>%
  slice(c(1:15, (n() - 14):n())) %>%
  mutate(
    direzione = ifelse(correlazione > 0,
                       "Correlazione positiva", "Correlazione negativa"),
    word = reorder(word, correlazione)
  ) %>%
  ggplot(aes(correlazione, word, fill = direzione)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  scale_fill_manual(values = c("Correlazione positiva" = "#2ecc71",
                               "Correlazione negativa" = "#e74c3c")) +
  labs(
    title    = "Termini più correlati al voto TMDB",
    subtitle = "Top 15 correlazioni positive e negative (DTM)",
    x = "Correlazione di Pearson", y = NULL, fill = NULL
  )

# --- 10b. Ridge / Lasso / Elastic Net ------------------------------------
set.seed(1234)
split   <- initial_split(bigdata, prop = 0.75)
train   <- training(split)
test    <- testing(split)

cols_out <- c("vote_average", "high_rated", "genre", "film_id", "title")
X_train  <- train %>% select(-any_of(cols_out)) %>% as.matrix()
X_test   <- test  %>% select(-any_of(cols_out)) %>% as.matrix()
y_train  <- train$vote_average
y_test   <- test$vote_average

set.seed(1234)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1,   nfolds = 10)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0,   nfolds = 10)
cv_enet  <- cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10)

pred_lasso <- predict(cv_lasso, newx = X_test, s = "lambda.min") %>% as.vector()
pred_ridge <- predict(cv_ridge, newx = X_test, s = "lambda.min") %>% as.vector()
pred_enet  <- predict(cv_enet,  newx = X_test, s = "lambda.min") %>% as.vector()

metriche_fn <- function(y_true, y_pred, nome) {
  tibble(
    Modello = nome,
    RMSE    = round(sqrt(mean((y_true - y_pred)^2)), 4),
    MAE     = round(mean(abs(y_true - y_pred)), 4),
    R2      = round(1 - sum((y_true - y_pred)^2) /
                      sum((y_true - mean(y_true))^2), 4)
  )
}

risultati_penalizzata <- bind_rows(
  metriche_fn(y_test, pred_lasso, "Lasso"),
  metriche_fn(y_test, pred_ridge, "Ridge"),
  metriche_fn(y_test, pred_enet,  "Elastic Net")
)
print(risultati_penalizzata)

# Predicted vs Actual (tutti e 3 i modelli)
tibble(
  actual       = y_test,
  Lasso        = pred_lasso,
  Ridge        = pred_ridge,
  `Elastic Net`= pred_enet
) %>%
  pivot_longer(-actual, names_to = "modello", values_to = "predicted") %>%
  ggplot(aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.25, size = 0.9, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_smooth(method = "lm", se = FALSE,
              color = "gray30", linewidth = 0.6, linetype = "dotted") +
  facet_wrap(~ modello) +
  labs(
    title    = "Predicted vs Actual – Modelli penalizzati",
    subtitle = "Linea rossa = predizione perfetta",
    x = "Voto reale", y = "Voto previsto"
  )

# Coefficienti Lasso
coef(cv_lasso, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  setNames("coeff") %>%                  # FIX: rinomina senza dipendere dal nome colonna
  rownames_to_column("word") %>%
  filter(coeff != 0, word != "(Intercept)") %>%
  arrange(desc(abs(coeff))) %>%
  slice_max(abs(coeff), n = 30) %>%
  mutate(
    direzione = ifelse(coeff > 0, "Aumenta il voto", "Diminuisce il voto"),
    word      = reorder(word, coeff)
  ) %>%
  ggplot(aes(coeff, word, fill = direzione)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  scale_fill_manual(values = c("Aumenta il voto"    = "#2ecc71",
                               "Diminuisce il voto" = "#e74c3c")) +
  labs(
    title    = "Parole selezionate dal Lasso (regressione)",
    subtitle = "Top 30 coefficienti non nulli",
    x = "Coefficiente", y = NULL, fill = NULL
  )


# =========================================================================
# 11. MODELLI PREDITTIVI – BOW + GLMNET + RANDOM FOREST
# =========================================================================

# Vocabolario: parole con frequenza >= 20 in tidy_overview
vocabolario <- tidy_overview %>%
  count(word, sort = TRUE) %>%
  filter(n >= 20) %>%
  pull(word)

# DTM sparsa (film_id × parola)
dtm_sparse <- tidy_overview %>%
  filter(word %in% vocabolario) %>%
  count(film_id, word) %>%
  cast_sparse(row = film_id, column = word, value = n)

# Etichette allineate
film_labels <- movies_clean %>%
  filter(film_id %in% as.integer(rownames(dtm_sparse))) %>%
  arrange(match(film_id, as.integer(rownames(dtm_sparse))))

y_class <- film_labels$high_rated     # factor Alto/Basso
y_reg   <- film_labels$vote_average

cat("\nDTM sparsa dimensioni:", nrow(dtm_sparse), "×", ncol(dtm_sparse), "\n")
cat("Distribuzione classi:\n"); print(table(y_class))

# --- Train/test split stratificato (80/20) --------------------------------
set.seed(42)
train_idx <- createDataPartition(y_class, p = 0.8, list = FALSE)

X_train_sp    <- dtm_sparse[ train_idx, ]
X_test_sp     <- dtm_sparse[-train_idx, ]
y_class_train <- y_class[ train_idx]
y_class_test  <- y_class[-train_idx]
y_reg_train   <- y_reg[   train_idx]
y_reg_test    <- y_reg[  -train_idx]

# ---- 11a. LASSO Logistico – classificazione ----------------------------
# FIX classe sbilanciata: usiamo penalty.factor per dare più peso alla
# classe minoritaria, oppure oversample.
# Qui usiamo il trucco weights proporzionale all'inverso della frequenza.

class_weights <- ifelse(
  y_class_train == "Alto",
  (length(y_class_train) / (2 * sum(y_class_train == "Alto"))),
  (length(y_class_train) / (2 * sum(y_class_train == "Basso")))
)

set.seed(42)
cv_class <- cv.glmnet(
  x            = X_train_sp,
  y            = y_class_train,
  family       = "binomial",
  alpha        = 1,
  nfolds       = 5,
  type.measure = "auc",              # AUC più informativa su classi sbilanciate
  weights      = class_weights
)

cat("\nLambda ottimale (classificazione AUC):", cv_class$lambda.min, "\n")
plot(cv_class, main = "CV – LASSO Logistico (AUC)")

pred_class_prob  <- predict(cv_class, newx = X_test_sp,
                            s = "lambda.min", type = "response") %>%
  as.vector()

# Soglia bilanciata: prevalenza nel training
soglia <- mean(y_class_train == "Alto")
pred_class_lasso <- factor(
  ifelse(pred_class_prob >= soglia, "Alto", "Basso"),
  levels = c("Alto", "Basso")
)

# Allineamento lunghezze (sicurezza)
n_min        <- min(length(pred_class_lasso), length(y_class_test))
pred_aligned <- pred_class_lasso[seq_len(n_min)]
true_aligned <- y_class_test[seq_len(n_min)]

cm_lasso <- confusionMatrix(pred_aligned, true_aligned, positive = "Alto")
print(cm_lasso)

# Curva ROC
roc_df <- tibble(
  prob  = pred_class_prob[seq_len(n_min)],
  label = true_aligned
) %>%
  arrange(desc(prob)) %>%
  mutate(
    tpr = cumsum(label == "Alto") / sum(label == "Alto"),
    fpr = cumsum(label == "Basso") / sum(label == "Basso")
  )

auc_val <- with(roc_df, sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1))) / 2)

ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#457B9D", linewidth = 1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray50") +
  annotate("text", x = 0.7, y = 0.2,
           label = sprintf("AUC = %.3f", auc_val), size = 5) +
  labs(
    title    = "Curva ROC – LASSO Logistico (classificazione)",
    subtitle = "Classe positiva: Alto (vote_average ≥ 7)",
    x = "False Positive Rate", y = "True Positive Rate"
  )

# Coefficienti LASSO classificazione
# FIX: setNames invece di rename(coef = s1)
coef(cv_class, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  setNames("coef") %>%
  rownames_to_column("word") %>%
  filter(word != "(Intercept)", coef != 0) %>%
  arrange(desc(abs(coef))) %>%
  slice_max(abs(coef), n = 25) %>%
  mutate(
    word      = reorder(word, coef),
    direzione = ifelse(coef > 0, "→ Alto (≥7)", "→ Basso (<7)")
  ) %>%
  ggplot(aes(coef, word, fill = direzione)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  scale_fill_manual(values = palette_class) +
  labs(
    title    = "LASSO: parole più discriminanti (high_rated)",
    subtitle = "Top 25 coefficienti non nulli",
    x = "Coefficiente logistico", y = NULL, fill = NULL
  )


# ---- 11b. Random Forest – classificazione --------------------------------

top_words_rf <- vocabolario[seq_len(min(300, length(vocabolario)))]

prep_rf <- function(mat, cols) {
  df <- as.matrix(mat[, colnames(mat) %in% cols]) %>% as.data.frame()
  missing <- setdiff(cols[cols %in% colnames(mat)], names(df))
  # aggiungi eventuali colonne mancanti con 0
  extra <- setdiff(
    cols[cols %in% colnames(as.matrix(dtm_sparse))],
    names(df)
  )
  df[extra] <- 0L
  df[, intersect(cols, names(df))]   # ordine coerente
}

X_train_rf <- as.matrix(X_train_sp[, colnames(X_train_sp) %in% top_words_rf]) %>%
  as.data.frame()
X_test_rf  <- as.matrix(X_test_sp[,  colnames(X_test_sp)  %in% top_words_rf]) %>%
  as.data.frame()

# colonne mancanti nel test → 0
miss <- setdiff(names(X_train_rf), names(X_test_rf))
X_test_rf[miss] <- 0L
X_test_rf <- X_test_rf[, names(X_train_rf)]

set.seed(42)
rf_class <- randomForest(
  x          = X_train_rf,
  y          = y_class_train,
  ntree      = 300,
  mtry       = floor(sqrt(ncol(X_train_rf))),
  classwt    = c("Alto" = 3, "Basso" = 1),   # FIX: peso classe minoritaria
  importance = TRUE
)

pred_rf_class <- predict(rf_class, newdata = X_test_rf)
cm_rf         <- confusionMatrix(pred_rf_class, y_class_test, positive = "Alto")
print(cm_rf)

# Variable importance
importance(rf_class, type = 1) %>%
  as.data.frame() %>%
  rownames_to_column("word") %>%
  slice_max(MeanDecreaseAccuracy, n = 20) %>%
  mutate(word = reorder(word, MeanDecreaseAccuracy)) %>%
  ggplot(aes(MeanDecreaseAccuracy, word, fill = MeanDecreaseAccuracy)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#a8d8b0", high = "#1a7c3e") +
  labs(
    title    = "Random Forest: importanza delle parole (classificazione)",
    subtitle = "Mean Decrease in Accuracy",
    x = "Mean Decrease Accuracy", y = NULL
  )


# ---- 11c. LASSO Lineare – regressione -----------------------------------

set.seed(42)
cv_reg <- cv.glmnet(
  x            = X_train_sp,
  y            = y_reg_train,
  family       = "gaussian",
  alpha        = 1,
  nfolds       = 5,
  type.measure = "mse"
)

plot(cv_reg, main = "CV – LASSO Lineare (MSE)")

pred_reg_lasso <- predict(cv_reg, newx = X_test_sp,
                          s = "lambda.min") %>% as.vector()

rmse_lasso_reg <- sqrt(mean((pred_reg_lasso - y_reg_test)^2))
mae_lasso_reg  <- mean(abs(pred_reg_lasso  - y_reg_test))
r2_lasso_reg   <- 1 - sum((pred_reg_lasso - y_reg_test)^2) /
  sum((y_reg_test - mean(y_reg_test))^2)

cat("\n--- LASSO Regressione (DTM sparsa) ---\n")
cat("RMSE:", round(rmse_lasso_reg, 4), "\n")
cat("MAE: ", round(mae_lasso_reg,  4), "\n")
cat("R²:  ", round(r2_lasso_reg,   4), "\n")

ggplot(data.frame(actual = y_reg_test, predicted = pred_reg_lasso),
       aes(actual, predicted)) +
  geom_point(alpha = 0.35, size = 0.9, color = "#457B9D") +
  geom_abline(slope = 1, intercept = 0,
              color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "lm", se = TRUE,
              color = "gray30", linewidth = 0.6) +
  labs(
    title    = "LASSO Regressione: Predicted vs Actual",
    subtitle = "Linea rossa = predizione perfetta | banda grigia = fit lineare",
    x = "Vote Average reale", y = "Vote Average predetto"
  )


# ---- 11d. Random Forest – regressione ------------------------------------

set.seed(42)
rf_reg <- randomForest(
  x          = X_train_rf,
  y          = y_reg_train,
  ntree      = 300,
  mtry       = floor(ncol(X_train_rf) / 3),
  importance = TRUE
)

pred_rf_reg <- predict(rf_reg, newdata = X_test_rf)

rmse_rf_reg <- sqrt(mean((pred_rf_reg - y_reg_test)^2))
mae_rf_reg  <- mean(abs(pred_rf_reg  - y_reg_test))
r2_rf_reg   <- 1 - sum((pred_rf_reg - y_reg_test)^2) /
  sum((y_reg_test - mean(y_reg_test))^2)

cat("\n--- Random Forest Regressione ---\n")
cat("RMSE:", round(rmse_rf_reg, 4), "\n")
cat("MAE: ", round(mae_rf_reg,  4), "\n")
cat("R²:  ", round(r2_rf_reg,   4), "\n")

ggplot(data.frame(actual = y_reg_test, predicted = pred_rf_reg),
       aes(actual, predicted)) +
  geom_point(alpha = 0.35, size = 0.9, color = "#2A9D8F") +
  geom_abline(slope = 1, intercept = 0,
              color = "red", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "lm", se = TRUE,
              color = "gray30", linewidth = 0.6) +
  labs(
    title    = "Random Forest Regressione: Predicted vs Actual",
    subtitle = "Linea rossa = predizione perfetta",
    x = "Vote Average reale", y = "Vote Average predetto"
  )


# =========================================================================
# 12. TABELLA RIASSUNTIVA FINALE
# =========================================================================

risultati_finali <- bind_rows(
  tibble(
    Modello  = c("LASSO Logistico", "Random Forest"),
    Task     = "Classificazione",
    Accuracy = c(cm_lasso$overall["Accuracy"],
                 cm_rf$overall["Accuracy"]),
    Sensitivity = c(cm_lasso$byClass["Sensitivity"],
                    cm_rf$byClass["Sensitivity"]),
    Specificity = c(cm_lasso$byClass["Specificity"],
                    cm_rf$byClass["Specificity"]),
    RMSE     = NA_real_,
    R2       = NA_real_
  ),
  tibble(
    Modello     = c("LASSO Lineare", "Random Forest"),
    Task        = "Regressione",
    Accuracy    = NA_real_,
    Sensitivity = NA_real_,
    Specificity = NA_real_,
    RMSE        = c(rmse_lasso_reg, rmse_rf_reg),
    R2          = c(r2_lasso_reg,   r2_rf_reg)
  )
) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

print(risultati_finali, n = Inf)






# =========================================================================
# 9b. ANALISI SUPERVISED LDA (sLDA)
# =========================================================================

cat("\n--- Inizio Addestramento Supervised LDA ---\n")

# Prepariamo la variabile di risposta (voto medio) allineata al corpus pulito
# Usiamo l'indice keep_docs per essere certi che i documenti corrispondano
y_slda <- movies_clean$vote_average[keep_docs]

# Parametri sLDA
K_slda <- 6       # Numero di topic
num_e_iter <- 100 # Iterazioni E-step (alza a 200+ per produzione)
num_m_iter <- 40  # Iterazioni M-step

# Inizializzazione dei coefficienti di regressione (pari a 0 per ogni topic)
initial_params <- rep(0, K_slda)

set.seed(12345)
slda_model <- slda.em(
  documents       = corpus_lda_clean,
  K               = K_slda,
  vocab           = to.keep.voc,
  num.e.iterations = num_e_iter,
  num.m.iterations = num_m_iter,
  alpha           = 0.1,
  eta             = 0.1,
  annotations     = y_slda,
  params          = initial_params,
  variance        = var(y_slda),
  logistic        = FALSE # Usiamo regressione lineare (Gaussian)
)

# --- Estrazione dei Topic Supervisionati ---------------------------------
top_words_slda <- top.topic.words(slda_model$topics, num.words = 8, by.score = TRUE)

# I coefficienti stimati per ciascun topic (l'effetto del topic sul voto)
topic_coefficients <- slda_model$coefs
names(topic_coefficients) <- paste("Topic", 1:K_slda)

cat("\nCoefficienti dei Topic (impatto sul voto):\n")
print(round(topic_coefficients, 4))

# Creiamo un dataframe per visualizzare le parole associate ai coefficienti
slda_topics_df <- tibble(
  topic = paste("Topic", 1:K_slda),
  coef  = topic_coefficients,
  top_words = apply(top_words_slda, 2, paste, collapse = ", ")
)

print(slda_topics_df)

# --- Grafico dei Coefficienti sLDA ---------------------------------------
slda_topics_df %>%
  mutate(
    topic = fct_reorder(topic, coef),
    direzione = ifelse(coef > 0, "Impatto Positivo (Alza il voto)", "Impatto Negativo (Abbassa il voto)")
  ) %>%
  ggplot(aes(x = coef, y = topic, fill = direzione)) +
  geom_col() +
  geom_text(aes(label = paste0("[", substr(top_words, 1, 30), "...]")), 
            hjust = ifelse(topic_coefficients > 0, -0.05, 1.05), size = 3) +
  scale_fill_manual(values = c("Impatto Positivo (Alza il voto)" = "#2ecc71", 
                               "Impatto Negativo (Abbassa il voto)" = "#e74c3c")) +
  theme_movies +
  labs(
    title = "Impatto dei Topic sLDA sul Voto dei Film",
    subtitle = "Coefficienti di regressione stimati congiuntamente al testo",
    x = "Coefficiente (Effetto sul Vote Average)", y = NULL, fill = NULL
  )

# --- Calcolo delle metriche predittive (In-Sample / Resoconto) ------------
# Nota: slda.em non ha una funzione nativa "predict" immediata per nuovi dati 
# nel pacchetto base senza ricalcolare i topic. Vediamo i valori stimati sul train:
predicted_ratings_slda <- slda_model$predictions

rmse_slda <- sqrt(mean((y_slda - predicted_ratings_slda)^2))
mae_slda  <- mean(abs(y_slda - predicted_ratings_slda))
r2_slda   <- 1 - sum((y_slda - predicted_ratings_slda)^2) / sum((y_slda - mean(y_slda))^2)

cat("\n--- Performance sLDA (Valutazione sul Fit) ---\n")
cat("RMSE:", round(rmse_slda, 4), "\n")
cat("MAE: ", round(mae_slda,  4), "\n")
cat("R²:  ", round(r2_slda,   4), "\n")