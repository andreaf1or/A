library(tidyverse)
library(ggrepel)
library(GGally)
library(lubridate)
library(scales)
library(patchwork)
library(naniar)
library(tm)
library(tidyverse)
library(tidytext)
library(wordcloud)
library(ngram)
library(lda)

movies <- read.csv("movies_dataset.csv")

movies <- movies %>% select(-backdrop_path, -poster_path, -original_title,
                            -popularity, -video, -id, -genre_ids) %>% 
  filter( release_date <= "2025-12-31") %>% 
  filter(vote_count >= 10)

# Analisi Esplorativa dei Dati
naniar::gg_miss_var(movies) +
  labs(title = "Missing Values in Movies Dataset",
       x = "Variables",
       y = "Count of Missing Values") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

movies <- movies %>% drop_na()

View(movies)





