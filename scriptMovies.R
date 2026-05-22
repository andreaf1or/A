# Librerie ----------------------------------------------------------------

library(tidyverse)
library(tidytext)
library(wordcloud)
library(ggwordcloud)
library(lubridate)
# Puoi rimuovere tm, lda, TextWiller, reshape2 se non le usi in questo script

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

data("stop_words")
parole_inutili <- tibble(word = c("film", "movie", "story", "life", "one", 
                                  "finds", "can", "will", "two", "new"))

# 2. Calcolo dei bigrammi
bigrammi <- movies_clean %>%
  # Estraiamo i bigrammi
  unnest_tokens(bigram, overview, token = "ngrams", n = 2) %>%
  # Separiamo in due colonne
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  # Filtriamo tutto ciò che non ci serve e gli NA
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word,
         !word1 %in% parole_inutili$word,
         !word2 %in% parole_inutili$word,
         !is.na(word1), !is.na(word2)) %>%
  # Riuniamo i termini puliti
  unite(bigram, word1, word2, sep = " ") %>%
  count(bigram, sort = TRUE)

# 3. Plot sicuro (senza rischio di grafico vuoto)
bigrammi %>%
  # Invece di filter(n > 60), prendiamo la Top 15
  slice_max(order_by = n, n = 15, with_ties = FALSE) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(x = n, y = bigram)) +
  geom_col(fill = "coral") +
  labs(title = "Bigrammi più frequenti nelle sinossi", 
       x = "Frequenza", 
       y = "")
# Wordcloud dei Bigrammi con ggwordcloud
bigrammi %>%
  # Ti consiglio di fermarti a 50 bigrammi, essendo frasi lunghe rubano molto spazio
  slice_max(order_by = n, n = 50, with_ties = FALSE) %>% 
  ggplot(aes(label = bigram, size = n, color = n)) +
  geom_text_wordcloud_area(shape = "square") + 
  scale_size_area(max_size = 10) +
  scale_color_gradient(low = "darkgray", high = "coral") + # Usa il coral del tuo grafico a barre
  theme_minimal() +
  labs(title = "Wordcloud dei Bigrammi più frequenti")
# Lunghezza sinossi per genere --------------------------------------------

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
