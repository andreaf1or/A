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
  drop_na() %>%
  mutate(
    year        = year(ymd(release_date)),
    genre_raw   = str_extract(genre_ids, "\\d+"),
    genre       = recode(genre_raw, !!!generi_lookup, .default = "Other"),
    high_rated  = as.factor(ifelse(vote_average >= 7, "Alto", "Basso")),
    n_parole    = str_count(overview, "\\S+"),
    n_caratteri = nchar(overview)
  ) %>%
  select(-genre_ids, -genre_raw, -release_date)


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
  filter(n > 400) %>%
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

# Contiamo le parole per ogni film
parole_per_film <- tidy_overview %>%
  count(id, word, sort = TRUE)

# teniamo i film che hanno almeno 1 parola
film_validi <- parole_per_film %>%
  group_by(id) %>%
  summarise(totale_parole = sum(n)) %>%
  filter(totale_parole > 0) %>%
  pull(id)

parole_per_film_filtrate <- parole_per_film %>%
  filter(id %in% film_validi)



# Creazione della DTM
movies_dtm <- parole_per_film_filtrate %>%
  cast_dtm(document = id, term = word, value = n)



# Stimiamo il modello
library(topicmodels)
set.seed(1234)
lda_model <- LDA(movies_dtm, k = 6, method = "Gibbs",
                 control = list(iter = 500))


# estrazione di Gamma
movie_topics <- tidy(lda_model, matrix = "gamma")

movie_topics_wide <- movie_topics %>%
  mutate(id = as.numeric(document)) %>% 
  select(-document) %>%
  pivot_wider(names_from = topic, names_prefix = "topic_", values_from = gamma)

# Ricreiamo il film_id sul dataset pulito originale per fare il join corretto
movies_con_voti_e_topic <- movies_clean %>%
  mutate(id = row_number()) %>% 
  inner_join(movie_topics_wide, by = "id")




movies_con_voti_e_topic %>%
  # Identifichiamo il topic con la probabilità più alta per ogni film
  pivot_longer(cols = starts_with("topic_"), names_to = "topic_prevalente", values_to = "probabilita") %>%
  group_by(id) %>%
  slice_max(probabilita, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  # Creiamo il grafico
  ggplot(aes(x = topic_prevalente, y = vote_average, fill = topic_prevalente)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  theme_minimal() +
  labs(
    title = "Analisi Pattern: Voto medio dei film per Topic LDA prevalente",
    subtitle = "I film sono raggruppati in base al tema principale della loro trama",
    x = "Topic Assegnato dall'Algoritmo",
    y = "Voto Medio (vote_average)"
  )



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

