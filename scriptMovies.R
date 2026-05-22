library(tidyverse)
library(tm)
library(tidytext)
library(wordcloud)
library(lda)
library(topicmodels)
library(TextWiller) 
library(reshape2)
library(wesanderson)
library(naniar)
library(lubridate)
library(ggwordcloud)

theme_set(theme_bw())

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
  # Rimuove colonne non utili all'analisi
  select(-backdrop_path, -poster_path, -original_title,
         -video, -adult) %>%
  # Filtri
  filter(release_date <= "2025-12-31",
         vote_count >= 10) %>%
  # Rimuove NA rimasti
  drop_na() %>%
  # Variabili derivate
  mutate(
    year        = year(ymd(release_date)),
    genre_raw   = str_extract(genre_ids, "\\d+"),
    genre       = recode(genre_raw, !!!generi_lookup, .default = "Other"),
    high_rated  = as.factor(ifelse(vote_average >= 7, "Alto", "Basso")),
    n_parole    = str_count(overview, "\\S+"),
    n_caratteri = nchar(overview)
  ) %>%
  # Ora che abbiamo genre possiamo togliere genre_ids e genre_raw
  select(-genre_ids, -genre_raw, -release_date)
glimpse(movies_clean)

# Preprocessing testo
movies_clean <- movies_clean %>%
  mutate(
    text_clean = overview %>%
      tolower() %>%
      removePunctuation() %>%
      removeNumbers() %>%
      (function(x) removeWords(x, stopwords("english"))) %>%
      stripWhitespace()
  )

glimpse(movies_clean %>% select(title, overview, text_clean, genre, vote_average))

# Analisi esplorativa
tidy_overview <- tibble(
  text  = movies_clean$text_clean,
  genre = movies_clean$genre,
  title = movies_clean$title,
  film  = 1:nrow(movies_clean)
) %>%
  unnest_tokens(word, text) %>%
  filter(nchar(word) > 3)
parole_inutili <- tibble(word = c("film", "movie", "story",
                                  "one", "finds", "can",
                                  "will", "two", "new"))
tidy_overview %>% count(word, sort = TRUE) %>% head(20)

tidy_overview %>%
  count(word, sort = TRUE) %>%
  filter(n > 400) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Parole più frequenti nelle trame dei film",
       x = "", y = "Frequenza")



# Ricalcoliamo le frequenze filtrando la spazzatura
word_counts_clean <- tidy_overview %>%
  anti_join(parole_inutili, by = "word") %>%
  count(word)

# Usiamo i dati puliti preparati nel passaggio precedente
word_counts_clean %>%
  # Prendiamo le 100 parole più frequenti per evitare il sovraffollamento
  slice_max(order_by = n, n = 100) %>% 
  ggplot(aes(label = word, size = n, color = n)) +
  # geom_text_wordcloud gestisce le collisioni e posiziona le parole
  geom_text_wordcloud(area_corr = TRUE) +
  # Controlla la dimensione massima delle parole
  scale_size_area(max_size = 15) +
  # Aggiungiamo un gradiente di colore in base alla frequenza
  scale_color_gradient(low = "darkgray", high = "firebrick") +
  theme_minimal() +
  labs(title = "Le parole più usate nelle trame")

