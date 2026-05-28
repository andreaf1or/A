# Librerie ----------------------------------------------------------------
library(glmnet)
library(tidyverse)
library(tidytext)
library(wordcloud)
library(ggwordcloud)
library(lubridate)
library(tm)
library(topicmodels)
library(rsample)
library(lda)
theme_set(theme_bw())

# Caricamento e pulizia dei dati ------------------------------------------

movies <- read_csv("movies_dataset.csv")

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
    high_rated  = as.factor(ifelse(vote_average >= 7, "Alto", "Basso")),
    n_parole    = str_count(overview, "\\S+"),
    n_caratteri = nchar(overview)
  ) %>%
  select(-genre_ids, -genre_raw, -release_date) %>% 
  drop_na()


# Preprocessing Tidy ------------------------------------------------------

# Definiamo le parole da escludere (combinando le tue con quelle di default inglesi)
data("stop_words")
parole_inutili <- tibble(word = c("film", "movie", "story", "life", "one",
                                  "finds", "can", "will", "two", "new"))

# Tokenizzazione e pulizia in un unico passaggio fluido
tidy_overview <- movies_clean %>%
  mutate(film_id = row_number()) %>% 
  unnest_tokens(word, overview) %>%
  # Rimuove numeri
  filter(!grepl("^[0-9]+$", word)) %>%
  # Rimuove parole corte
  filter(nchar(word) > 3) %>%
  # Rimuove stopwords inglesi di default
  anti_join(stop_words, by = "word") %>%
  # Rimuove le tue parole custom
  anti_join(parole_inutili, by = "word")


# Analisi Esplorativa -----------------------------------------------------

# Conteggio generale pulito
word_counts_clean <- tidy_overview %>% count(word, sort = TRUE)

# Grafico a barre
word_counts_clean %>%
  filter(n > 200) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Parole più frequenti nelle trame dei film", x = "", y = "Frequenza")

# Wordcloud con ggwordcloud
word_counts_clean %>%
  slice_max(order_by = n, n = 100) %>% 
  ggplot(aes(label = word, size = n, color = n)) +
  geom_text_wordcloud(area_corr = TRUE) +
  scale_size_area(max_size = 15) +
  scale_color_gradient(low = "darkgray", high = "firebrick") +
  theme_minimal() +
  labs(title = "Le parole più usate nelle trame")

# Wordcloud per genere (multi-plot)
generi_principali <- c("Action", "Drama", "Comedy", "Horror", "Thriller", "Romance")
# Filtriamo i dati per i generi principali e calcoliamo le frequenze
word_counts_generi <- tidy_overview %>%
  filter(genre %in% generi_principali) %>%
  count(genre, word, sort = TRUE) %>%
  # Limite fondamentale: prendiamo solo le 30 parole principali per ogni genere
  group_by(genre) %>%
  slice_max(order_by = n, n = 30, with_ties = FALSE) %>%
  ungroup()

# Creiamo il multi-plot con ggwordcloud
word_counts_generi %>%
  ggplot(aes(label = word, size = n, color = genre)) +
  # geom_text_wordcloud gestisce gli spazi stringendo le parole
  geom_text_wordcloud_area(shape = "circle") + 
  # Se le parole ti sembrano ancora troppo grandi o tagliate, abbassa max_size
  scale_size_area(max_size = 10) + 
  # facet_wrap divide automaticamente il grafico in riquadri per genere
  facet_wrap(~ genre, ncol = 3) +
  theme_minimal() +
  theme(
    strip.text = element_text(size = 12, face = "bold", color = "black"),
    strip.background = element_rect(fill = "lightgray", color = NA)
  ) +
  labs(title = "Wordcloud delle trame per Genere",
       subtitle = "Le 30 parole più frequenti")

# Bigrammi -------------------------------------------------------

parole_inutili <- tibble(word = c("film", "movie", "story", "life", "one", 
                                  "finds", "can", "will", "two", "new", "_"))
# Calcolo dei bigrammi
bigrammi <- movies_clean %>%
  unnest_tokens(bigram, overview, token = "ngrams", n = 2) %>%
  # Separiamo in due colonne
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word1 %in% parole_inutili$word,
         !word2 %in% parole_inutili$word,
         !is.na(word1), !is.na(word2)) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE)

bigrammi %>%
  # Top 15 bigrammi per evitare errori di visualizzazione
  slice_max(order_by = n, n = 15, with_ties = FALSE) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(x = n, y = bigram)) +
  geom_col(fill = "coral") +
  labs(title = "Bigrammi più frequenti nelle sinossi", 
       x = "Frequenza", 
       y = "")

# Wordcloud dei Bigrammi con ggwordcloud primi 50 bigrammi
bigrammi %>%
  slice_max(order_by = n, n = 50, with_ties = FALSE) %>% 
  ggplot(aes(label = bigram, size = n, color = n)) +
  geom_text_wordcloud_area(shape = "square") + 
  scale_size_area(max_size = 10) +
  scale_color_gradient(low = "darkgray", high = "coral") + # Usa il coral del tuo grafico a barre
  theme_minimal() +
  labs(title = "Wordcloud dei Bigrammi più frequenti")


# Lunghezza trame per genere --------------------------------------------

movies_clean %>%
  pivot_longer(cols = c(n_parole, n_caratteri),
               names_to  = "metrica",
               values_to = "valore") %>%
  ggplot(aes(x = reorder(genre, valore, FUN = median), y = valore, fill = genre)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  facet_wrap(~ metrica, scales = "free_x",
             labeller = as_labeller(c(n_caratteri = "Numero di Caratteri",
                                      n_parole    = "Numero di Parole"))) +
  coord_flip() +
  theme(legend.position = "none", strip.text = element_text(face = "bold")) +
  labs(title = "Lunghezza delle sinossi per genere", x = "", y = NULL)


# TF-IDF ---------------------------------------------------------

tfidf_genre <- tidy_overview %>%
  filter(!is.na(genre), !is.na(word)) %>% 
  count(genre, word, sort = TRUE) %>%
  # Argomenti esplicitati per evitare valori negativi
  bind_tf_idf(term = word, document = genre, n = n)


tfidf_genre %>%
  group_by(genre) %>%
  # with_ties = FALSE per avere esattamente 5 parole per genere e nessun errore grafico
  slice_max(tf_idf, n = 5, with_ties = FALSE) %>% 
  ungroup() %>%
  mutate(word = reorder_within(word, tf_idf, genre)) %>%
  ggplot(aes(tf_idf, word, fill = genre)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ genre, scales = "free_y", ncol = 4) +
  scale_y_reordered() +
  labs(title = "Parole più distintive per genere (TF-IDF)", x = "TF-IDF", y = "")


## Analisi LDA ------------------------------------------------------

N <- nrow(movies_clean)
K <- 6
I <- 100

corpus <- lexicalize(tidy_overview$word, lower = T)
n_voc <- word.counts(corpus$documents, corpus$vocab)
hist(n_voc, breaks = 100)

data.frame(v = corpus$vocab, n = n_voc) %>% arrange(-n)

plot(ecdf(n_voc))
mean(n_voc == 1)
mean(n_voc <= 5)

# includiamo la parola se appare almeno 5 volte
to.keep.voc <- corpus$vocab[n_voc >= 5]
corpus <- lexicalize(tidy_overview$word,
                     vocab = to.keep.voc)


# Algoritmo di Gibbs Sampling
set.seed(12345)
result <- lda.collapsed.gibbs.sampler(documents = corpus,
                                      K = K,
                                      vocab = to.keep.voc,
                                      num.iterations = I,
                                      alpha = 0.1,
                                      eta = 0.1,
                                      burnin = 100,
                                      compute.log.likelihood = T)

matplot(t(result$log.likelihood), type = "l")

# le 6 parole più rappresentative e importani (riga) per ogni topic (colonna)
top.words <- top.topic.words(result$topics, num.words = 6, by.score = T)
top.words



corpus <- VectorSource(movies_clean$overview)
dtm <- DocumentTermMatrix(corpus,
                          control = list(stopwords = "english",
                                         removeNumbers = T)) %>% 
  removeSparseTerms(0.995)
temp <- as.matrix(dtm)

n.parole <- NA
n.caratteri <- NA
for(i in 1:nrow(movie_clean)){
  n.parole[i] <- wordcount
}



################################################

movies_topics_words <- tidy(lda_model, matrix = "beta")

# Selezioniamo le prime 10 parole più importanti per ogni topic
top_words_per_topic <- movies_topics_words %>%
  group_by(topic) %>%
  slice_max(beta, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic))

# Facciamo il grafico a barre speculare a quello del laboratorio
ggplot(top_words_per_topic, aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 3) +
  scale_y_reordered() +
  theme_minimal() +
  labs(
    title = "Termini più rilevanti per ogni Topic (Matrice Beta)",
    x = "Beta (Probabilità del termine nel topic)",
    y = NULL
  )


##############################

movies_con_voti_e_topic %>%
  pivot_longer(cols = starts_with("topic_"), names_to = "topic", values_to = "valore_gamma") %>%
  ggplot(aes(x = valore_gamma, y = vote_average)) +
  geom_point(alpha = 0.05, color = "darkblue") + # Mostra la nuvola dei film
  geom_smooth(method = "lm", color = "firebrick", se = FALSE) + # Linea di tendenza
  facet_wrap(~ topic, ncol = 3) +
  theme_minimal() +
  labs(
    title = "Impatto dei Temi delle Trame sul Voto del Pubblico",
    x = "Rilevanza del Topic nel film (Gamma)",
    y = "Voto Medio del film (vote_average)"
  )


# Costruzione DTM ---------------------------------------------------------


# Costruiamo il corpus dalla colonna text_clean
# (se non l'avete, usate overview direttamente)
corpus <- movies_clean %>%
  pull(overview) %>%
  VectorSource() %>%
  VCorpus() %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(content_transformer(function(x)
    iconv(x, to = "ASCII//TRANSLIT"))) %>%   
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removeWords, parole_inutili$word) %>%
  tm_map(stripWhitespace)

# DTM con removeSparseTerms — identico al Lab9
dtm <- DocumentTermMatrix(corpus) %>%
  removeSparseTerms(0.99)  # tiene termini presenti in almeno 1% dei film

cat("DTM dimensioni:", dim(dtm), "\n")  # film x termini

bigdata <- as.matrix(dtm) %>%
  as.data.frame() %>%
  mutate(
    vote_average = movies_clean$vote_average,
    high_rated   = movies_clean$high_rated,
    genre        = movies_clean$genre,
    film_id      = 1:nrow(movies_clean)
  )
dim(bigdata)
glimpse(bigdata)

# Calcolo correlazioni tra termini e voto medi
correlazioni <- dtm %>%
  as.matrix() %>%
  as.data.frame() %>%
  mutate(vote_average = movies_clean$vote_average) %>%
  summarise(across(-vote_average,
                   ~ cor(.x, vote_average, use = "complete.obs"))) %>%
  pivot_longer(everything(), names_to = "word", values_to = "correlazione") %>%
  arrange(desc(abs(correlazione)))

correlazioni %>%
  slice(c(1:15, (n()-14):n())) %>%
  mutate(
    direzione = ifelse(correlazione > 0, "Correlazione positiva", "Correlazione negativa"),
    word = reorder(word, correlazione)
  ) %>%
  ggplot(aes(correlazione, word, fill = direzione)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("Correlazione positiva" = "#2ecc71",
                               "Correlazione negativa" = "#e74c3c")) +
  labs(title = "Termini più correlati al voto TMDB (dalla DTM)",
       x = "Correlazione di Pearson", y = "", fill = "")


# Regressione Ridge, Lasso ed Elastic Net penalizzata ---------------------

set.seed(1234)
split <- initial_split(bigdata)
train <- training(split)
test  <- testing(split)

# X esclude le colonne non numeriche e il target
# y è vote_average
cols_da_escludere <- c("vote_average", "high_rated", "genre", "film_id", "title")

X_train <- train %>% select(-any_of(cols_da_escludere)) %>% as.matrix()
X_test  <- test  %>% select(-any_of(cols_da_escludere)) %>% as.matrix()
y_train <- train$vote_average
y_test  <- test$vote_average

# Lasso
set.seed(1234)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1,   nfolds = 10)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0,   nfolds = 10)
cv_enet  <- cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10)

# Predict sul test
pred_lasso <- predict(cv_lasso, newx = X_test, s = "lambda.min") %>% as.vector()
pred_ridge <- predict(cv_ridge, newx = X_test, s = "lambda.min") %>% as.vector()
pred_enet  <- predict(cv_enet,  newx = X_test, s = "lambda.min") %>% as.vector()

# Metriche
metriche <- function(y_true, y_pred, nome) {
  rmse <- sqrt(mean((y_true - y_pred)^2))
  r2   <- 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  tibble(Modello = nome, RMSE = round(rmse, 4), R2 = round(r2, 4))
}

risultati <- bind_rows(
  metriche(y_test, pred_lasso, "Lasso"),
  metriche(y_test, pred_ridge, "Ridge"),
  metriche(y_test, pred_enet,  "Elastic Net")
)
print(risultati)

# Predicted vs Actual
tibble(
  actual = y_test,
  Lasso  = pred_lasso,
  Ridge  = pred_ridge,
  `Elastic Net` = pred_enet
) %>%
  pivot_longer(-actual, names_to = "modello", values_to = "predicted") %>%
  ggplot(aes(x = actual, y = predicted, colour = modello)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey30") +
  facet_wrap(~ modello) +
  theme(legend.position = "none") +
  labs(title = "Predicted vs Actual sul Test Set",
       x = "Voto reale", y = "Voto previsto")

coef(cv_lasso, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("word") %>%
  rename(coeff = 2) %>%       
  filter(coeff != 0, word != "(Intercept)") %>%
  arrange(desc(abs(coeff))) %>%
  mutate(
    direzione = ifelse(coeff > 0, "Aumenta il voto", "Diminuisce il voto"),
    word = reorder(word, coeff)
  ) %>%
  ggplot(aes(coeff, word, fill = direzione)) +
  geom_col() +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_fill_manual(values = c("Aumenta il voto"    = "#2ecc71",
                               "Diminuisce il voto" = "#e74c3c")) +
  labs(title = "Parole selezionate dal Lasso",
       x = "Coefficiente", y = "", fill = "")

# ============================================================
# MODELLI PREDITTIVI - BOW + GLMNET + RANDOM FOREST
# ============================================================
library(glmnet)
library(randomForest)
library(caret)
library(Matrix)

# ------------------------------------------------------------
# 1. COSTRUZIONE DELLA DTM (Bag of Words)
# ------------------------------------------------------------

# Vocabolario: teniamo solo le parole con frequenza >= 20
# per ridurre dimensionalità ed evitare rumore
vocabolario <- tidy_overview %>%
  count(word, sort = TRUE) %>%
  filter(n >= 20) %>%
  pull(word)

# Costruiamo la Document-Term Matrix sparsa
# ogni riga = film, ogni colonna = parola, cella = frequenza
dtm_sparse <- tidy_overview %>%
  filter(word %in% vocabolario) %>%
  count(film_id, word) %>%
  cast_sparse(row = film_id, column = word, value = n)

# Recuperiamo le etichette target allineate ai film_id rimasti nella DTM
film_labels <- movies_clean %>%
  mutate(film_id = row_number()) %>%
  filter(film_id %in% as.integer(rownames(dtm_sparse))) %>%
  arrange(match(film_id, as.integer(rownames(dtm_sparse))))

# Vettori target
y_class <- film_labels$high_rated          # factor "Alto"/"Basso" → classificazione
y_reg   <- film_labels$vote_average        # numerico              → regressione

cat("Dimensione DTM:", nrow(dtm_sparse), "film x", ncol(dtm_sparse), "parole\n")
cat("Distribuzione classi:\n"); print(table(y_class))


# ------------------------------------------------------------
# 2. TRAIN / TEST SPLIT (80/20 stratificato)
# ------------------------------------------------------------

set.seed(42)
train_idx <- createDataPartition(y_class, p = 0.8, list = FALSE)

X_train <- dtm_sparse[ train_idx, ]
X_test  <- dtm_sparse[-train_idx, ]

y_class_train <- y_class[ train_idx]
y_class_test  <- y_class[-train_idx]
y_reg_train   <- y_reg[   train_idx]
y_reg_test    <- y_reg[  -train_idx]


# ============================================================
# 3. CLASSIFICAZIONE: high_rated (Alto / Basso)
# ============================================================

# --- 3a. LASSO LOGISTICO (glmnet, alpha=1) ------------------

set.seed(42)
cv_class <- cv.glmnet(
  x      = X_train,
  y      = y_class_train,
  family = "binomial",
  alpha  = 1,          # LASSO: seleziona automaticamente le parole utili
  nfolds = 5,
  type.measure = "class"
)

# Lambda ottimale
lambda_class <- cv_class$lambda.min
cat("Lambda ottimale (classificazione):", lambda_class, "\n")

# Plot curva CV
plot(cv_class, main = "CV - LASSO Logistico (Classificazione)")

# Predizione sul test set
pred_class_lasso <- predict(cv_class,
                            newx   = X_test,
                            s      = "lambda.min",
                            type   = "class") %>% as.vector()

# Matrice di confusione e metriche
cm_lasso <- confusionMatrix(
  factor(pred_class_lasso, levels = c("Alto", "Basso")),
  y_class_test
)
print(cm_lasso)

# Parole più importanti selezionate dal LASSO
coef_lasso <- coef(cv_class, s = "lambda.min") %>%
  as.matrix() %>%
  as.data.frame() %>%
  rownames_to_column("word") %>%
  rename(coef = s1) %>%
  filter(word != "(Intercept)", coef != 0) %>%
  arrange(desc(abs(coef)))

# Plot: top 20 parole discriminanti
coef_lasso %>%
  slice_max(abs(coef), n = 20) %>%
  mutate(
    word      = reorder(word, coef),
    direzione = ifelse(coef > 0, "→ Alto (≥7)", "→ Basso (<7)")
  ) %>%
  ggplot(aes(coef, word, fill = direzione)) +
  geom_col() +
  scale_fill_manual(values = c("→ Alto (≥7)" = "steelblue",
                               "→ Basso (<7)" = "coral")) +
  labs(title = "LASSO: parole più discriminanti per high_rated",
       x = "Coefficiente", y = "", fill = "")


# --- 3b. RANDOM FOREST (classificazione) --------------------
# RF non accetta matrici sparse → convertiamo in data.frame denso
# Per evitare problemi di memoria, limitiamo a top 300 parole

top_words_class <- vocabolario[1:min(300, length(vocabolario))]

X_train_rf <- as.matrix(X_train[, colnames(X_train) %in% top_words_class]) %>%
  as.data.frame()
X_test_rf  <- as.matrix(X_test[,  colnames(X_test)  %in% top_words_class]) %>%
  as.data.frame()

# Allineiamo le colonne (alcune potrebbero mancare nel test)
missing_cols <- setdiff(names(X_train_rf), names(X_test_rf))
X_test_rf[missing_cols] <- 0
X_test_rf <- X_test_rf[, names(X_train_rf)]

set.seed(42)
rf_class <- randomForest(
  x         = X_train_rf,
  y         = y_class_train,
  ntree     = 300,
  mtry      = floor(sqrt(ncol(X_train_rf))),
  importance = TRUE
)

pred_rf_class <- predict(rf_class, newdata = X_test_rf)

cm_rf <- confusionMatrix(pred_rf_class, y_class_test)
print(cm_rf)

# Variable importance RF
importance(rf_class, type = 1) %>%
  as.data.frame() %>%
  rownames_to_column("word") %>%
  rename(MeanDecreaseAccuracy = MeanDecreaseAccuracy) %>%
  slice_max(MeanDecreaseAccuracy, n = 20) %>%
  mutate(word = reorder(word, MeanDecreaseAccuracy)) %>%
  ggplot(aes(MeanDecreaseAccuracy, word)) +
  geom_col(fill = "forestgreen") +
  labs(title = "Random Forest: top 20 parole per importanza",
       x = "Mean Decrease Accuracy", y = "")


# ============================================================
# 4. REGRESSIONE: vote_average (score continuo)
# ============================================================

# --- 4a. LASSO LINEARE (glmnet) -----------------------------

set.seed(42)
cv_reg <- cv.glmnet(
  x      = X_train,
  y      = y_reg_train,
  family = "gaussian",
  alpha  = 1,
  nfolds = 5,
  type.measure = "mse"
)

plot(cv_reg, main = "CV - LASSO Lineare (Regressione)")

pred_reg_lasso <- predict(cv_reg,
                          newx = X_test,
                          s    = "lambda.min") %>% as.vector()

# Metriche regressione
rmse_lasso <- sqrt(mean((pred_reg_lasso - y_reg_test)^2))
mae_lasso  <- mean(abs(pred_reg_lasso  - y_reg_test))
r2_lasso   <- 1 - sum((pred_reg_lasso - y_reg_test)^2) /
  sum((y_reg_test - mean(y_reg_test))^2)

cat("\n--- LASSO Regressione ---\n")
cat("RMSE:", round(rmse_lasso, 4), "\n")
cat("MAE: ", round(mae_lasso,  4), "\n")
cat("R²:  ", round(r2_lasso,   4), "\n")

# Predicted vs Actual
data.frame(actual = y_reg_test, predicted = pred_reg_lasso) %>%
  ggplot(aes(actual, predicted)) +
  geom_point(alpha = 0.4, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "LASSO Regressione: Predicted vs Actual",
       x = "Vote Average reale", y = "Vote Average predetto")


# --- 4b. RANDOM FOREST (regressione) ------------------------

set.seed(42)
rf_reg <- randomForest(
  x         = X_train_rf,
  y         = y_reg_train,
  ntree     = 300,
  mtry      = floor(ncol(X_train_rf) / 3),
  importance = TRUE
)

pred_rf_reg <- predict(rf_reg, newdata = X_test_rf)

rmse_rf <- sqrt(mean((pred_rf_reg - y_reg_test)^2))
mae_rf  <- mean(abs(pred_rf_reg  - y_reg_test))
r2_rf   <- 1 - sum((pred_rf_reg - y_reg_test)^2) /
  sum((y_reg_test - mean(y_reg_test))^2)

cat("\n--- Random Forest Regressione ---\n")
cat("RMSE:", round(rmse_rf, 4), "\n")
cat("MAE: ", round(mae_rf,  4), "\n")
cat("R²:  ", round(r2_rf,   4), "\n")

data.frame(actual = y_reg_test, predicted = pred_rf_reg) %>%
  ggplot(aes(actual, predicted)) +
  geom_point(alpha = 0.4, color = "forestgreen") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Random Forest Regressione: Predicted vs Actual",
       x = "Vote Average reale", y = "Vote Average predetto")


# ============================================================
# 5. CONFRONTO RIASSUNTIVO DEI MODELLI
# ============================================================

risultati <- tibble(
  Modello  = c("LASSO", "Random Forest"),
  Task     = c("Classificazione", "Classificazione"),
  Accuracy = c(cm_lasso$overall["Accuracy"],
               cm_rf$overall["Accuracy"]),
) %>%
  bind_rows(
    tibble(
      Modello = c("LASSO", "Random Forest"),
      Task    = c("Regressione", "Regressione"),
      RMSE    = c(rmse_lasso, rmse_rf),
      R2      = c(r2_lasso,   r2_rf)
    )
  )

print(risultati)



##### Classificazione di testi 

corpus <- VectorSource(movies_clean$overview)
dtm <- DocumentTermMatrix(corpus,
                          control = list(stopwords = "english")) %>% 
  removeSparseTerms(.9)
bigdata <- data.frame(as.matrix(dtm))





