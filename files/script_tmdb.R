# =========================================================================
# PROGETTO: ANALISI DELLE TRAME E PREDIZIONE DEL VOTO DEI FILM
# =========================================================================

# -------------------------------------------------------------------------
# 1. SETUP E LIBRERIE
# -------------------------------------------------------------------------
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
library(ggrepel)

# --- IMPOSTAZIONE DEL SEME GLOBALE ---
# Usiamo questa costante prima di ogni operazione stocastica
GLOBAL_SEED <- 12345

# -------------------------------------------------------------------------
# 2. TEMA GRAFICO GLOBALE E PALETTE
# -------------------------------------------------------------------------
colori_topic  <- c("#E63946", "#457B9D", "#2A9D8F", 
                   "#E9C46A", "#F4A261", "#264653")
palette_class <- c("Alto (>= 7)" = "#457B9D", "Basso (< 7)" = "#E63946")

theme_movies <- theme_bw(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 14),
    plot.subtitle    = element_text(color = "gray40", size = 11),
    axis.title       = element_text(size = 11),
    strip.text       = element_text(face = "bold"),
    strip.background = element_rect(fill = "gray92", color = NA),
    legend.position  = "bottom"
  )
theme_set(theme_movies)

# -------------------------------------------------------------------------
# 3. CARICAMENTO E PULIZIA DEI DATI
# -------------------------------------------------------------------------
# Assicurati che il file "movies_dataset.csv" sia nella tua working directory
movies <- read_csv("movies_dataset.csv", show_col_types = FALSE)

generi_lookup <- c(
  "28"    = "Action",    "12"    = "Adventure", "16"    = "Animation",
  "35"    = "Comedy",    "80"    = "Crime",     "99"    = "Documentary",
  "18"    = "Drama",     "10751" = "Family",    "14"    = "Fantasy",
  "36"    = "History",   "27"    = "Horror",    "10402" = "Music",
  "9648"  = "Mystery",   "10749" = "Romance",   "878"   = "Sci-Fi",
  "53"    = "Thriller",  "10752" = "War",       "37"    = "Western"
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
  mutate(film_id = row_number())

# -------------------------------------------------------------------------
# 4. PREPROCESSING DEL TESTO E TOKENIZZAZIONE
# -------------------------------------------------------------------------
text_preprocessing <- function(x) {
  x <- gsub("http\\S+\\s*",            "",  x)
  x <- gsub("#\\S+",                   "",  x)
  x <- gsub("[[:cntrl:]]",             "",  x)
  x <- gsub("^[[:space:]]*|[[:space:]]*$", "", x)
  x <- gsub(" +",                      " ", x)
  x
}

parole_inutili <- tibble(word = c(
  "film", "movie", "story", "life", "one", "finds", "can", "will",
  "two",  "new",   "world", "_",    "set", "takes", "make", "time",
  "back", "love",  "family"
))

# Prep testo base
movies_clean$overview <- gsub("\\b\\d{1,2}x\\d{2}\\b", "", movies_clean$overview) %>%
  tolower() %>%
  removeWords(stopwords("SMART")) %>%
  text_preprocessing() %>%
  removePunctuation() %>%
  removeNumbers() %>%
  stripWhitespace() %>%
  removeWords(stopwords("SMART"))

data("stop_words")

# Tokenizzazione
tidy_overview <- movies_clean %>%
  select(film_id, genre, overview) %>%
  unnest_tokens(output = word, input = overview) %>%
  anti_join(stop_words,     by = "word") %>%
  anti_join(parole_inutili, by = "word") %>%
  filter(nchar(word) > 2)

word_counts <- tidy_overview %>% count(word, sort = TRUE)

# -------------------------------------------------------------------------
# 5. ANALISI ESPLORATIVA GRAFICA
# -------------------------------------------------------------------------
# 5a. Distribuzione voti
movies_clean %>%
  mutate(classe = ifelse(vote_average >= 7, "Alto (>= 7)", "Basso (< 7)")) %>%
  ggplot(aes(x = vote_average, fill = classe)) +
  geom_histogram(binwidth = 0.5, color = "white", alpha = 0.85) +
  scale_fill_manual(values = palette_class) +
  labs(title = "Distribuzione dei Voti (Vote Average)", x = "Voto Medio", y = "Conteggio")
ggsave("hist_voti.png", width = 8, height = 5)

# 5b. Boxplot per genere
movies_clean %>%
  mutate(genre = fct_reorder(genre, vote_average, .fun = median)) %>%
  ggplot(aes(x = genre, y = vote_average, fill = genre)) +
  geom_boxplot(show.legend = FALSE, alpha = 0.7) +
  geom_hline(yintercept = 7, linetype = "dashed", color = "#E63946", linewidth = 0.8) +
  coord_flip() +
  labs(title = "Voto per Genere", x = "Genere", y = "Voto Medio")
ggsave("votixgenere.png", width = 9, height = 6)

# 5c. Wordcloud
set.seed(GLOBAL_SEED) # Seme per il layout della wordcloud
word_counts %>%
  slice_max(order_by = n, n = 100) %>%
  ggplot(aes(label = word, size = n, color = n)) +
  geom_text_wordcloud(area_corr = TRUE, rm_outside = TRUE) +
  scale_size_area(max_size = 14) +
  scale_color_gradient(low = "gray60", high = "#c0392b") +
  theme_minimal() +
  labs(title = "Wordcloud delle trame")
ggsave("wordcloud1.png", width = 8, height = 6)

# 5d. Bigrammi
movies_clean %>%
  unnest_tokens(bigram, overview, token = "ngrams", n = 2) %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word, !word2 %in% stop_words$word,
         !word1 %in% parole_inutili$word, !word2 %in% parole_inutili$word,
         !is.na(word1), !is.na(word2), nchar(word1) > 2, nchar(word2) > 2) %>%
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE) %>%
  slice_max(n, n = 20) %>%
  mutate(bigram = fct_reorder(bigram, n)) %>%
  ggplot(aes(x = n, y = bigram, fill = n)) +
  geom_col(show.legend = FALSE) +
  scale_fill_gradient(low = "#f4a261", high = "#c0392b") +
  labs(title = "Top 20 Bigrammi", x = "Frequenza", y = NULL)
ggsave("bigrams.png", width = 8, height = 6)

# -------------------------------------------------------------------------
# 6. TOPIC MODELING SUPERVISIONATO (sLDA)
# -------------------------------------------------------------------------
testi_per_film <- tidy_overview %>%
  group_by(film_id) %>%
  summarise(testo = paste(word, collapse = " "), .groups = "drop")

corpus_slda_full <- lexicalize(testi_per_film$testo, lower = TRUE)
n_voc            <- word.counts(corpus_slda_full$documents, corpus_slda_full$vocab)
to.keep.voc      <- corpus_slda_full$vocab[n_voc >= 5]
corpus_slda      <- lexicalize(testi_per_film$testo, vocab = to.keep.voc, lower = TRUE)

keep_docs <- which(sapply(corpus_slda, function(d) if (is.null(d) || !is.matrix(d)) 0L else ncol(d)) > 0)
y_slda    <- movies_clean$vote_average[keep_docs]

K_slda <- 6
set.seed(GLOBAL_SEED) # Seme per l'algoritmo EM di sLDA
slda_model <- slda.em(
  documents = corpus_slda, K = K_slda, vocab = to.keep.voc,
  num.e.iterations = 100, num.m.iterations = 40,
  alpha = 0.1, eta = 0.1, annotations = y_slda, params = rep(0, K_slda),
  variance = var(y_slda), logistic = FALSE
)

topic.names_slda <- c("Dramma Familiare & Relazioni", "Crime & Poliziesco", 
                      "Commedia Romantica & Teen", "Horror & Mystery", 
                      "Storico, Biografie & Drama", "Fantascienza & Fantasy")

# -------------------------------------------------------------------------
# 7. REGRESSIONE PENALIZZATA (DTM Densa)
# -------------------------------------------------------------------------
corpus_tm <- movies_clean$overview %>% VectorSource() %>% VCorpus() %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(content_transformer(function(x) iconv(x, to = "ASCII//TRANSLIT", sub = ""))) %>%
  tm_map(removePunctuation) %>% tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(removeWords, parole_inutili$word) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus_tm) %>% removeSparseTerms(0.99) %>% as.matrix() %>% as.data.frame()
dtm$vote_average <- movies_clean$vote_average # FIX FONDAMENTALE

set.seed(GLOBAL_SEED) # Seme per lo split dei dati
split <- initial_split(dtm, prop = 0.75)
train <- training(split)
test  <- testing(split)

X_train <- train %>% select(-vote_average) %>% as.matrix()
X_test  <- test %>% select(-vote_average) %>% as.matrix()
y_train <- train$vote_average
y_test  <- test$vote_average

# Semi separati per la Cross-Validation interna di glmnet
set.seed(GLOBAL_SEED)
cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, nfolds = 10)
set.seed(GLOBAL_SEED)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10)
set.seed(GLOBAL_SEED)
cv_enet  <- cv.glmnet(X_train, y_train, alpha = 0.5, nfolds = 10)

pred_lasso <- predict(cv_lasso, newx = X_test, s = "lambda.min") %>% as.vector()
pred_ridge <- predict(cv_ridge, newx = X_test, s = "lambda.min") %>% as.vector()
pred_enet  <- predict(cv_enet,  newx = X_test, s = "lambda.min") %>% as.vector()

# -------------------------------------------------------------------------
# 8. MODELLI PREDITTIVI SU DTM SPARSA
# -------------------------------------------------------------------------
vocabolario <- tidy_overview %>% count(word) %>% filter(n >= 20) %>% pull(word)

dtm_sparse <- tidy_overview %>%
  filter(word %in% vocabolario) %>%
  count(film_id, word) %>%
  cast_sparse(row = film_id, column = word, value = n)

film_labels <- movies_clean %>% filter(film_id %in% as.integer(rownames(dtm_sparse))) %>% arrange(match(film_id, as.integer(rownames(dtm_sparse))))
y_class <- film_labels$high_rated
y_reg   <- film_labels$vote_average

set.seed(GLOBAL_SEED) # Seme per la partizione stratificata
train_idx <- createDataPartition(y_class, p = 0.8, list = FALSE)

X_train_sp <- dtm_sparse[train_idx, ]
X_test_sp  <- dtm_sparse[-train_idx, ]
y_class_train <- y_class[train_idx]
y_class_test  <- y_class[-train_idx]
y_reg_train   <- y_reg[train_idx]
y_reg_test    <- y_reg[-train_idx]

# --- 8a. LASSO Logistico (Classificazione) ---
class_weights <- ifelse(y_class_train == "Alto", 
                        length(y_class_train) / (2 * sum(y_class_train == "Alto")),
                        length(y_class_train) / (2 * sum(y_class_train == "Basso")))

set.seed(GLOBAL_SEED) # Seme per CV Lasso Logistico
cv_class <- cv.glmnet(X_train_sp, y_class_train, family = "binomial", alpha = 1, nfolds = 5, type.measure = "auc", weights = class_weights)

pred_class_prob <- predict(cv_class, newx = X_test_sp, s = "lambda.min", type = "response") %>% as.vector()
soglia <- mean(y_class_train == "Alto")
pred_class_lasso <- factor(ifelse(pred_class_prob >= soglia, "Alto", "Basso"), levels = c("Alto", "Basso"))
n_min <- min(length(pred_class_lasso), length(y_class_test))
cm_lasso <- confusionMatrix(pred_class_lasso[seq_len(n_min)], y_class_test[seq_len(n_min)], positive = "Alto")

# --- 8b. Random Forest (Classificazione) ---
top_words_rf <- vocabolario[seq_len(min(300, length(vocabolario)))]
X_train_rf <- as.data.frame(as.matrix(X_train_sp[, colnames(X_train_sp) %in% top_words_rf]))
X_test_rf  <- as.data.frame(as.matrix(X_test_sp[, colnames(X_test_sp) %in% top_words_rf]))
miss <- setdiff(names(X_train_rf), names(X_test_rf))
X_test_rf[miss] <- 0L
X_test_rf <- X_test_rf[, names(X_train_rf)]

set.seed(GLOBAL_SEED) # Seme per Random Forest
rf_class <- randomForest(x = X_train_rf, y = y_class_train, ntree = 300, mtry = floor(sqrt(ncol(X_train_rf))), classwt = c("Alto"=3, "Basso"=1))
cm_rf <- confusionMatrix(predict(rf_class, newdata = X_test_rf), y_class_test, positive = "Alto")

# --- 8c. Modelli di Regressione (Sparsa) ---
set.seed(GLOBAL_SEED)
cv_reg <- cv.glmnet(X_train_sp, y_reg_train, family = "gaussian", alpha = 1, nfolds = 5)
pred_reg_lasso <- predict(cv_reg, newx = X_test_sp, s = "lambda.min") %>% as.vector()

set.seed(GLOBAL_SEED)
rf_reg <- randomForest(x = X_train_rf, y = y_reg_train, ntree = 300, mtry = floor(ncol(X_train_rf)/3))
pred_rf_reg <- predict(rf_reg, newdata = X_test_rf)

# -------------------------------------------------------------------------
# 9. RISULTATI FINALI
# -------------------------------------------------------------------------
rmse_lasso_reg <- sqrt(mean((pred_reg_lasso - y_reg_test)^2))
r2_lasso_reg   <- 1 - sum((pred_reg_lasso - y_reg_test)^2) / sum((y_reg_test - mean(y_reg_test))^2)
rmse_rf_reg    <- sqrt(mean((pred_rf_reg - y_reg_test)^2))
r2_rf_reg      <- 1 - sum((pred_rf_reg - y_reg_test)^2) / sum((y_reg_test - mean(y_reg_test))^2)

risultati_finali <- bind_rows(
  tibble(Modello = "LASSO Logistico", Task = "Classificazione", Accuracy = cm_lasso$overall["Accuracy"], Sensitivity = cm_lasso$byClass["Sensitivity"], Specificity = cm_lasso$byClass["Specificity"], RMSE = NA_real_, R2 = NA_real_),
  tibble(Modello = "Random Forest Class", Task = "Classificazione", Accuracy = cm_rf$overall["Accuracy"], Sensitivity = cm_rf$byClass["Sensitivity"], Specificity = cm_rf$byClass["Specificity"], RMSE = NA_real_, R2 = NA_real_),
  tibble(Modello = "LASSO Lineare", Task = "Regressione", Accuracy = NA_real_, Sensitivity = NA_real_, Specificity = NA_real_, RMSE = rmse_lasso_reg, R2 = r2_lasso_reg),
  tibble(Modello = "Random Forest Reg", Task = "Regressione", Accuracy = NA_real_, Sensitivity = NA_real_, Specificity = NA_real_, RMSE = rmse_rf_reg, R2 = r2_rf_reg)
) %>% mutate(across(where(is.numeric), ~ round(.x, 4)))

cat("\n=== Tabella Riassuntiva Finale ===\n")
print(risultati_finali)

