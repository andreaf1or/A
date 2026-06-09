install.packages("devtools")
install.packages("lifecycle")
devtools::install_github("lchiffon/wordcloud2")
# PreProcessing -----------------------------------------------------------
text_preprocessing <- function(x) {
  x <- gsub('http\\S+\\s*','',x)         # remove URLs
  x <- gsub('#\\S+','',x)                # remove hashtags
  x <- gsub('[[:cntrl:]]','',x)          # remove controls and special characters
  x <- gsub("^[[:space:]]*","",x)        # remove leading whitespaces
  x <- gsub("[[:space:]]*$","",x)        # remove trailing whitespaces
  x <- gsub(' +', ' ', x)                # remove extra whitespaces
  return(x)                              # restituiamo il testo pulito
}
x <- movies_clean$overview
x_clean <- gsub("\\b\\d{1,2}x\\d{2}\\b", "", x) %>%  
  tolower() %>% 
  removeWords(words = stopwords("SMART")) %>%  
  text_preprocessing() %>% 
  removePunctuation() %>% 
  removeNumbers() %>% 
  stripWhitespace() %>% 
  removeWords(words = stopwords("SMART"))
movies_clean$overview <- x_clean

tidy_movies <- movies_clean %>% 
  select(overview) %>% 
  unnest_tokens(output = word, input = overview)
data("stop_words")
parole_inutili <- tibble(word = c("film", "movie", "story", "life", "one",
                                  "finds", "can", "will", "two", "new"))
tidy_movies <- tidy_movies %>% 
  anti_join(stop_words, by = 'word') %>% 
  anti_join(parole_inutili, by = 'word')
  
word_counts <- tidy_movies %>% count(word, sort = T)

word_counts %>% slice_max(n, n = 50) %>% 
  ggplot()+
  geom_col(aes(x = n, y=fct_reorder(word, n)))
sfumature_grigio <- colorRampPalette(c("black", "gray80"))(100)
wordcloud2::wordcloud2(word_counts %>% slice_max(n, n=100) %>% rename(freq = n),
                       backgroundColor = 'white',
                       figPath = 'camera.png', size = .4,
                       color = sfumature_grigio)

