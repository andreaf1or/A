# =========================================================================
# PROGETTO: ANALISI DELLE TRAME E PREDIZIONE DEL VOTO DEI FILM
# =========================================================================

# 0. Librerie e Setup -----------------------------------------------------
library(tidyverse)
library(tidytext)
library(ggwordcloud)
library(lubridate)
library(tm)
library(topicmodels)
library(rsample)
library(glmnet)
library(randomForest)
library(caret)
library(Matrix)
library(yardstick)
# Impostiamo il tema globale per i grafici
theme_set(theme_bw())

# 1. Caricamento e Pulizia dei Dati ---------------------------------------
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
  drop_na(overview, release_date, vote_average, genre_ids) %>%
  mutate(
    # Creiamo un ID univoco fin dall'inizio, fondamentale per l'ML
    film_id     = row_number(), 
    year        = year(ymd(release_date)),
    genre_raw   = str_extract(genre_ids, "\\d+"),
    genre       = recode(genre_raw, !!!generi_lookup, .default = "Other"),
    high_rated  = as.factor(ifelse(vote_average >= 6.5, "Alto", "Basso")),
    n_parole    = str_count(overview, "\\S+"),
    n_caratteri = nchar(overview)
  ) %>%
  select(-genre_ids, -genre_raw, -release_date)


# 2. Preprocessing Tidy (Testo) -------------------------------------------
data("stop_words")
parole_inutili <- tibble(word = c("film", "movie", "story", "life", "one",
                                  "finds", "can", "will", "two", "new"))

tidy_overview <- movies_clean %>%
  unnest_tokens(word, overview) %>%
  filter(!grepl("^[0-9]+$", word)) %>% # Rimuove numeri
  filter(nchar(word) > 3) %>%          # Rimuove parole corte
  anti_join(stop_words, by = "word") %>%
  anti_join(parole_inutili, by = "word")


# 3. Analisi Esplorativa (EDA) --------------------------------------------

# 3.1 Parole più frequenti
word_counts_clean <- tidy_overview %>% count(word, sort = TRUE)

word_counts_clean %>%
  filter(n > 400) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Parole più frequenti nelle trame dei film", x = "", y = "Frequenza")

# 3.2 Wordcloud generale
set.seed(123)
word_counts_clean %>%
  slice_max(order_by = n, n = 100) %>% 
  ggplot(aes(label = word, size = n, color = n)) +
  geom_text_wordcloud(area_corr = TRUE) +
  scale_size_area(max_size = 15) +
  scale_color_gradient(low = "darkgray", high = "firebrick") +
  theme_minimal() +
  labs(title = "Le parole più usate nelle trame")

# 3.3 Wordcloud per genere
generi_principali <- c("Action", "Drama", "Comedy", "Horror", "Thriller", "Romance")

set.seed(123)
tidy_overview %>%
  filter(genre %in% generi_principali) %>%
  count(genre, word, sort = TRUE) %>%
  group_by(genre) %>%
  slice_max(order_by = n, n = 30, with_ties = FALSE) %>%
  ungroup() %>%
  ggplot(aes(label = word, size = n, color = genre)) +
  geom_text_wordcloud_area(shape = "circle") + 
  scale_size_area(max_size = 8) + 
  facet_wrap(~ genre, ncol = 3) +
  theme_minimal() +
  theme(strip.text = element_text(size = 12, face = "bold", color = "black"),
        strip.background = element_rect(fill = "lightgray", color = NA)) +
  labs(title = "Wordcloud delle trame per Genere")

# 3.4 Bigrammi
bigrammi <- movies_clean %>%
  unnest_tokens(bigram, overview, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word, !word2 %in% stop_words$word,
         !word1 %in% parole_inutili$word, !word2 %in% parole_inutili$word,
         !is.na(word1), !is.na(word2)) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE)

bigrammi %>%
  slice_max(order_by = n, n = 15, with_ties = FALSE) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(x = n, y = bigram)) +
  geom_col(fill = "coral") +
  labs(title = "Bigrammi più frequenti", x = "Frequenza", y = "")

# 3.5 TF-IDF (Parole distintive per genere)
tidy_overview %>%
  filter(!is.na(genre), !is.na(word)) %>% 
  count(genre, word, sort = TRUE) %>%
  bind_tf_idf(term = word, document = genre, n = n) %>%
  group_by(genre) %>%
  slice_max(tf_idf, n = 5, with_ties = FALSE) %>% 
  ungroup() %>%
  mutate(word = reorder_within(word, tf_idf, genre)) %>%
  ggplot(aes(tf_idf, word, fill = genre)) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ genre, scales = "free_y", ncol = 4) +
  scale_y_reordered() +
  labs(title = "Parole più distintive per genere (TF-IDF)", x = "TF-IDF", y = "")


# 4. Analisi LDA (Topic Modeling) -----------------------------------------
cat("\nAvvio estrazione Topics (LDA)...\n")

# Creiamo la DTM specifica per l'algoritmo LDA
dtm_lda <- tidy_overview %>%
  count(film_id, word) %>%
  cast_dtm(document = film_id, term = word, value = n)

# Modello LDA
set.seed(12345)
lda_model <- LDA(dtm_lda, k = 6, control = list(seed = 12345))

# Plot Matrice Beta (Parole per Topic)
tidy(lda_model, matrix = "beta") %>%
  group_by(topic) %>%
  slice_max(beta, n = 10, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free", ncol = 3) +
  scale_y_reordered() +
  theme_minimal() +
  labs(title = "Termini più rilevanti per ogni Topic (Matrice Beta)", y = NULL)

# Plot Matrice Gamma (Impatto Topic sul Voto)
document_topic_matrix <- tidy(lda_model, matrix = "gamma") %>%
  mutate(topic = paste0("topic_", topic)) %>%
  pivot_wider(names_from = topic, values_from = gamma) %>%
  mutate(document = as.integer(document)) 

movies_con_voti_e_topic <- movies_clean %>%
  inner_join(document_topic_matrix, by = c("film_id" = "document"))

movies_con_voti_e_topic %>%
  pivot_longer(cols = starts_with("topic_"), names_to = "topic", values_to = "valore_gamma") %>%
  ggplot(aes(x = valore_gamma, y = vote_average)) +
  geom_point(alpha = 0.05, color = "darkblue") +
  geom_smooth(method = "lm", color = "firebrick", se = FALSE) +
  facet_wrap(~ topic, ncol = 3) +
  theme_minimal() +
  labs(title = "Impatto dei Temi delle Trame sul Voto del Pubblico",
       x = "Rilevanza del Topic nel film (Gamma)", y = "Voto Medio")


# 5. Machine Learning - Preparazione Dati ---------------------------------
cat("\nPreparazione dati per Machine Learning...\n")

# Teniamo solo le parole apparse almeno 20 volte per ridurre il rumore
vocabolario <- tidy_overview %>%
  count(word, sort = TRUE) %>%
  filter(n >= 20) %>%
  pull(word)

# Creazione Matrice Sparsa (Veloce e leggera in RAM)
dtm_sparse <- tidy_overview %>%
  filter(word %in% vocabolario) %>%
  count(film_id, word) %>%
  cast_sparse(row = film_id, column = word, value = n)

# Assicuriamoci che etichette e matrice siano allineate
film_labels <- movies_clean %>%
  filter(film_id %in% as.integer(rownames(dtm_sparse))) %>%
  arrange(match(film_id, as.integer(rownames(dtm_sparse))))

y_class <- film_labels$high_rated  # Per classificazione (Alto/Basso)
y_reg   <- film_labels$vote_average # Per regressione (Numerico)

# Train/Test Split (80/20)
set.seed(42)
train_idx <- createDataPartition(y_class, p = 0.8, list = FALSE)

X_train <- dtm_sparse[train_idx, ]
X_test  <- dtm_sparse[-train_idx, ]

y_class_train <- y_class[train_idx]
y_class_test  <- y_class[-train_idx]
y_reg_train   <- y_reg[train_idx]
y_reg_test    <- y_reg[-train_idx]


# 6. Modelli di Classificazione (Alto vs Basso) ---------------------------
cat("\nAddestramento LASSO Classificazione...\n")
set.seed(42)

# Assicuriamoci che i livelli siano ordinati con la classe di interesse ("Alto") per prima
y_class_train <- factor(y_class_train, levels = c("Alto", "Basso"))
y_class_test  <- factor(y_class_test, levels = c("Alto", "Basso"))

cv_class_lasso <- cv.glmnet(x = X_train, y = y_class_train, family = "binomial", 
                            alpha = 1, nfolds = 5, type.measure = "class")

# Predizione CLASSI per Matrice di Confusione
pred_class_lasso <- predict(cv_class_lasso, newx = X_test, s = "lambda.min", type = "class") %>% as.vector()
cm_lasso <- confusionMatrix(factor(pred_class_lasso, levels = c("Alto", "Basso")), y_class_test)

# Predizione PROBABILITA' per Curve ROC/Lift (glmnet modella il secondo livello, quindi 1 - prob)
prob_lasso <- 1 - (predict(cv_class_lasso, newx = X_test, s = "lambda.min", type = "response") %>% as.vector())

# Plot Parole selezionate dal Lasso per la classificazione
coef(cv_class_lasso, s = "lambda.min") %>%
  as.matrix() %>% 
  as.data.frame() %>% 
  rownames_to_column("word") %>% 
  rename(coef = 2) %>% 
  filter(word != "(Intercept)", coef != 0) %>% 
  arrange(desc(abs(coef))) %>%
  slice_max(abs(coef), n = 20) %>%
  mutate(word = reorder(word, coef), 
         direzione = ifelse(coef > 0, "Sopra la media (≥6.5)", "Sotto la media (<6.5)")) %>%
  ggplot(aes(coef, word, fill = direzione)) +
  geom_col() +
  scale_fill_manual(values = c("Sopra la media (≥6.5)" = "steelblue", 
                               "Sotto la media (<6.5)" = "coral")) +
  labs(title = "LASSO: parole più discriminanti per Voto", x = "Coefficiente", y = "", fill = "")


# Random Forest per Classificazione 
top_words_ml <- vocabolario[1:min(300, length(vocabolario))]
X_train_rf <- as.matrix(X_train[, colnames(X_train) %in% top_words_ml]) %>% as.data.frame()
X_test_rf  <- as.matrix(X_test[, colnames(X_test) %in% top_words_ml]) %>% as.data.frame()
missing_cols <- setdiff(names(X_train_rf), names(X_test_rf))
X_test_rf[missing_cols] <- 0
X_test_rf <- X_test_rf[, names(X_train_rf)] # Riordina colonne

cat("\nAddestramento Random Forest Classificazione...\n")
set.seed(42)
rf_class <- randomForest(x = X_train_rf, y = y_class_train, ntree = 150, 
                         mtry = floor(sqrt(ncol(X_train_rf))), importance = TRUE)

# Predizione CLASSI e PROBABILITA' per Random Forest
pred_rf_class <- predict(rf_class, newdata = X_test_rf)
cm_rf <- confusionMatrix(pred_rf_class, y_class_test)
prob_rf <- predict(rf_class, newdata = X_test_rf, type = "prob")[, "Alto"]

# --- CREAZIONE CURVE ROC E LIFT CON YARDSTICK ---

# Combiniamo i risultati in un unico dataset pulito per yardstick
df_valutazione <- tibble(
  truth = y_class_test,
  LASSO = prob_lasso,
  RandomForest = prob_rf
) %>%
  pivot_longer(cols = c(LASSO, RandomForest), names_to = "Modello", values_to = "Probabilita")

# Plot Curva ROC
plot_roc <- df_valutazione %>%
  group_by(Modello) %>%
  roc_curve(truth, Probabilita) %>%
  autoplot() +
  labs(title = "Curva ROC - Predizione film 'Alto'") +
  scale_color_manual(values = c("LASSO" = "steelblue", "RandomForest" = "forestgreen")) +
  theme_minimal()

print(plot_roc)

# Plot Curva Lift
plot_lift <- df_valutazione %>%
  group_by(Modello) %>%
  lift_curve(truth, Probabilita) %>%
  autoplot() +
  labs(title = "Curva Lift - Predizione film 'Alto'",
       subtitle = "Mostra quanto è più efficace il modello rispetto a pescare a caso") +
  scale_color_manual(values = c("LASSO" = "steelblue", "RandomForest" = "forestgreen")) +
  theme_minimal()

print(plot_lift)

# Calcolo esatto dell'AUC per la tabella finale
auc_lasso <- df_valutazione %>% filter(Modello == "LASSO") %>% roc_auc(truth, Probabilita) %>% pull(.estimate)
auc_rf    <- df_valutazione %>% filter(Modello == "RandomForest") %>% roc_auc(truth, Probabilita) %>% pull(.estimate)


# 8. Confronto Riassuntivo Finale -----------------------------------------
risultati_finali <- tibble(
  Modello  = c("LASSO", "Random Forest", "LASSO", "Random Forest"),
  Task     = c("Classificazione", "Classificazione", "Regressione", "Regressione"),
  Metrica = c(paste0("Accuracy: ", round(cm_lasso$overall["Accuracy"]*100, 2), "% | AUC: ", round(auc_lasso, 3)),
              paste0("Accuracy: ", round(cm_rf$overall["Accuracy"]*100, 2), "% | AUC: ", round(auc_rf, 3)),
              paste0("RMSE: ", round(rmse_lasso, 3)),
              paste0("RMSE: ", round(rmse_rf, 3)))
)

cat("\n================ CONFRONTO FINALE =================\n")
print(risultati_finali)
