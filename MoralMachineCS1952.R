install.packages(c("httr", "jsonlite", "cluster", "factoextra", "reticulate"))

library(httr)
library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(countrycode)
library(cluster)
library(factoextra)
library(reticulate)

# PART 1: Download and Load Data

download_moral_machine_data <- function(max_rows = 20000, batch_size = 100, 
                                        resume_file = "download_progress.rds",
                                        delay_seconds = 1,
                                        rate_limit_delay = 60) {
  base_url <- "https://datasets-server.huggingface.co/rows"
  dataset <- "Sonya3/MoralMachineHuman"
  
  # Check if we have a saved progress file
  if(file.exists(resume_file)) {
    cat("Found previous download progress. Loading...\n")
    saved_data <- readRDS(resume_file)
    all_data <- saved_data$all_data
    offset <- saved_data$offset
    cat("Resuming from offset", offset, "\n")
  } else {
    all_data <- list()
    offset <- 0
  }
  
  consecutive_errors <- 0
  max_retries <- 5
  
  while(offset < max_rows) {
    cat("Downloading rows", offset, "to", offset + batch_size, "\n")
    
    tryCatch({
      response <- GET(
        base_url,
        query = list(
          dataset = dataset,
          config = "default",
          split = "train",
          offset = offset,
          length = batch_size
        ),
        timeout(30)
      )
      
      if(status_code(response) == 200) {
        batch_data <- fromJSON(content(response, "text", encoding = "UTF-8"))
        
        if(length(batch_data$rows) == 0) {
          cat("No more data available. Download complete.\n")
          break
        }
        
        all_data[[length(all_data) + 1]] <- batch_data$rows
        offset <- offset + batch_size
        consecutive_errors <- 0
        
        # Save progress every 1000 rows
        if(offset %% 1000 == 0) {
          saveRDS(list(all_data = all_data, offset = offset), resume_file)
          cat("Progress saved at offset", offset, "\n")
        }
        
        Sys.sleep(delay_seconds)
        
      } else if(status_code(response) == 429) {
        # Rate limit hit - wait longer before retry
        cat("Rate limit (429) hit at offset", offset, 
            ". Waiting", rate_limit_delay, "seconds...\n")
        consecutive_errors <- consecutive_errors + 1
        
        if(consecutive_errors >= max_retries) {
          cat("Too many rate limit errors. Stopping.\n")
          break
        }
        
        Sys.sleep(rate_limit_delay)
        next  # Don't increment offset, try again
        
      } else {
        cat("HTTP error", status_code(response), "at offset", offset, "\n")
        consecutive_errors <- consecutive_errors + 1
        
        if(consecutive_errors >= max_retries) {
          cat("Too many consecutive errors. Stopping.\n")
          break
        }
        
        Sys.sleep(5)
        next
      }
      
    }, error = function(e) {
      cat("Error at offset", offset, ":", conditionMessage(e), "\n")
      consecutive_errors <<- consecutive_errors + 1
      
      if(consecutive_errors >= max_retries) {
        cat("Too many consecutive errors. Stopping.\n")
        return(NULL)
      }
      
      Sys.sleep(5)
    })
    
    if(consecutive_errors >= max_retries) break
  }
  
  # Final save
  saveRDS(list(all_data = all_data, offset = offset), resume_file)
  
  # Combine all batches
  if(length(all_data) > 0) {
    combined_data <- bind_rows(lapply(all_data, function(x) {
      as.data.frame(x$row)
    }))
    
    cat("\nDownload complete. Total rows:", nrow(combined_data), "\n")
    
    return(combined_data)
  } else {
    cat("No data downloaded.\n")
    return(NULL)
  }
}

# Resume download with longer delays to avoid rate limiting
moral_machine <- download_moral_machine_data(
  max_rows = 20000, 
  batch_size = 100,
  delay_seconds = 2,        # Wait 2 seconds between successful requests
  rate_limit_delay = 60     # Wait 60 seconds when rate limited
)

# Save the data
if(!is.null(moral_machine)) {
  saveRDS(moral_machine, "moral_machine_data.rds")
  cat("Data saved to moral_machine_data.rds\n")
  cat("Total rows downloaded:", nrow(moral_machine), "\n")
}

# let's load the actual data
moral_machine <- readRDS("moral_machine_data.rds")

# Basic data structure
str(moral_machine)
head(moral_machine)
summary(moral_machine)

# Check for missing values
colSums(is.na(moral_machine))

# Country distribution
country_counts <- moral_machine %>%
  count(Country, sort = TRUE)

print(country_counts)

# Response distribution
response_counts <- moral_machine %>%
  count(HumanResponse, sort = TRUE)

print(response_counts)

# Check unique values in key columns -- IMPORTANT FIGURES FOR THE PRESENTATION
cat("\nUnique Countries:", length(unique(moral_machine$Country)), "\n")
cat("Unique Responses:", length(unique(moral_machine$HumanResponse)), "\n")
cat("Total Observations:", nrow(moral_machine), "\n")

# CLEAN 

# Remove rows with missing Country
moral_machine_clean <- moral_machine %>%
  filter(!is.na(Country))

cat("Rows after removing missing Country:", nrow(moral_machine_clean), "\n")

# Add a column to categorize the decision type
moral_machine_clean <- moral_machine_clean %>%
  mutate(
    # Check if question mentions passengers/car occupants vs pedestrians
    mentions_passengers = grepl("sitting in the car", Question, ignore.case = TRUE),
    mentions_barrier = grepl("hit a barrier", Question, ignore.case = TRUE),
    
    # Classify decision: if response mentions typical passenger descriptions
    # This is a simplified heuristic - responses show WHO to save
    decision = case_when(
      grepl("sitting in the car", Question) ~ "passenger_vs_pedestrian",
      grepl("crossing the street from the left side.*crossing the street from the right side", Question) ~ "pedestrian_vs_pedestrian",
      TRUE ~ "other"
    )
  )

table(moral_machine_clean$decision)


# PART 3: Extract Features from Questions

extract_features_from_text <- function(text) {
  data.frame(
    has_elderly = grepl("elderly", text, ignore.case = TRUE),
    has_child = grepl("\\b(boy|girl|child|baby)\\b", text, ignore.case = TRUE),
    has_male = grepl("\\b(man|men|male)\\b", text, ignore.case = TRUE),
    has_female = grepl("\\b(woman|women|female)\\b", text, ignore.case = TRUE),
    has_athlete = grepl("athlete", text, ignore.case = TRUE),
    has_large = grepl("large", text, ignore.case = TRUE),
    has_doctor = grepl("doctor", text, ignore.case = TRUE),
    has_executive = grepl("executive", text, ignore.case = TRUE),
    has_homeless = grepl("homeless", text, ignore.case = TRUE),
    has_criminal = grepl("criminal", text, ignore.case = TRUE),
    has_pet = grepl("\\b(dog|cat)\\b", text, ignore.case = TRUE),
    lawful = grepl("abiding by the law|green signal", text, ignore.case = TRUE),
    unlawful = grepl("flouting the law|red signal", text, ignore.case = TRUE)
  )
}

# Extract features from questions
question_features <- extract_features_from_text(moral_machine_clean$Question)
colnames(question_features) <- paste0("q_", colnames(question_features))

# Extract features from responses (who was saved)
response_features <- extract_features_from_text(moral_machine_clean$HumanResponse)
colnames(response_features) <- paste0("r_", colnames(response_features))

# Combine
moral_machine_features <- bind_cols(moral_machine_clean, question_features, response_features)

# PART 4: Country-Level Analysis

# Calculate response preferences by country
country_summary <- moral_machine_features %>%
  group_by(Country) %>%
  summarise(
    n = n(),
    # Who people chose to save
    pct_saved_elderly = mean(r_has_elderly, na.rm = TRUE) * 100,
    pct_saved_children = mean(r_has_child, na.rm = TRUE) * 100,
    pct_saved_athletes = mean(r_has_athlete, na.rm = TRUE) * 100,
    pct_saved_large = mean(r_has_large, na.rm = TRUE) * 100,
    pct_saved_doctors = mean(r_has_doctor, na.rm = TRUE) * 100,
    pct_saved_executives = mean(r_has_executive, na.rm = TRUE) * 100,
    pct_saved_criminals = mean(r_has_criminal, na.rm = TRUE) * 100,
    pct_saved_pets = mean(r_has_pet, na.rm = TRUE) * 100,
    .groups = 'drop'
  ) %>%
  filter(n >= 70)  # Keep countries with at least 70 responses

# Add region
country_summary$Region <- countrycode(
  country_summary$Country, 
  origin = "iso3c", 
  destination = "continent",
  warn = FALSE
)

print(country_summary)

# SCENARIO ANALYSIS BY COUNTRY AND REGION

# PART 1: Identify key moral dimensions from the data

# Analyze what characteristics appear in questions and responses
moral_dimensions <- moral_machine_features %>%
  group_by(Country) %>%
  summarise(
    n = n(),
    # When these groups appear in scenarios, how often are they saved?
    save_elderly_rate = sum(r_has_elderly, na.rm = TRUE) / sum(q_has_elderly, na.rm = TRUE) * 100,
    save_children_rate = sum(r_has_child, na.rm = TRUE) / sum(q_has_child, na.rm = TRUE) * 100,
    save_athletes_rate = sum(r_has_athlete, na.rm = TRUE) / sum(q_has_athlete, na.rm = TRUE) * 100,
    save_large_rate = sum(r_has_large, na.rm = TRUE) / sum(q_has_large, na.rm = TRUE) * 100,
    save_doctors_rate = sum(r_has_doctor, na.rm = TRUE) / sum(q_has_doctor, na.rm = TRUE) * 100,
    save_executives_rate = sum(r_has_executive, na.rm = TRUE) / sum(q_has_executive, na.rm = TRUE) * 100,
    save_criminals_rate = sum(r_has_criminal, na.rm = TRUE) / sum(q_has_criminal, na.rm = TRUE) * 100,
    save_pets_rate = sum(r_has_pet, na.rm = TRUE) / sum(q_has_pet, na.rm = TRUE) * 100,
    save_female_rate = sum(r_has_female, na.rm = TRUE) / sum(q_has_female, na.rm = TRUE) * 100,
    save_male_rate = sum(r_has_male, na.rm = TRUE) / sum(q_has_male, na.rm = TRUE) * 100,
    .groups = 'drop'
  ) %>%
  filter(n >= 100)

# Add region
moral_dimensions$Region <- countrycode(
  moral_dimensions$Country, 
  origin = "iso3c", 
  destination = "continent",
  warn = FALSE
)

# PART 2: Regional aggregation

regional_patterns <- moral_dimensions %>%
  filter(!is.na(Region)) %>%
  group_by(Region) %>%
  summarise(
    n_countries = n_distinct(Country),
    n_responses = sum(n),
    across(ends_with("_rate"), ~mean(.x, na.rm = TRUE)),
    .groups = 'drop'
  )

print(regional_patterns)

# PART 3: VISUALIZATIONS

# 1. Top 20 countries - who do they save?
top_countries <- moral_dimensions %>%
  arrange(desc(n)) %>%
  head(20) %>%
  select(Country, n, ends_with("_rate")) %>%
  pivot_longer(cols = ends_with("_rate"), names_to = "dimension", values_to = "save_rate") %>%
  mutate(dimension = gsub("save_|_rate", "", dimension))

ggplot(top_countries, aes(x = dimension, y = save_rate, fill = Country)) +
  geom_col(position = "dodge") +
  facet_wrap(~Country, ncol = 4) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        legend.position = "none") +
  labs(title = "Saving Preferences by Top 20 Countries",
       x = "Group Type", y = "% Saved (when present in scenario)")

ggsave("top20_country_patterns.pdf", width = 16, height = 12)

# 2. Regional comparison - key dimensions
regional_long <- regional_patterns %>%
  select(Region, ends_with("_rate")) %>%
  pivot_longer(cols = ends_with("_rate"), names_to = "dimension", values_to = "save_rate") %>%
  mutate(dimension = gsub("save_|_rate", "", dimension))

ggplot(regional_long, aes(x = reorder(dimension, save_rate), y = save_rate, fill = Region)) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(title = "Regional Patterns in Moral Preferences",
       subtitle = "% of times each group is saved when present in scenario",
       x = "Group", y = "Save Rate (%)") +
  theme_minimal() +
  theme(legend.position = "bottom")

ggsave("regional_patterns.pdf", width = 12, height = 8)

# 3. Age preference: Children vs Elderly by country
age_preferences <- moral_dimensions %>%
  arrange(desc(n)) %>%
  head(30) %>%
  select(Country, Region, save_children_rate, save_elderly_rate) %>%
  mutate(child_preference = save_children_rate - save_elderly_rate)

ggplot(age_preferences, aes(x = save_elderly_rate, y = save_children_rate, 
                            color = Region, label = Country)) +
  geom_point(size = 3) +
  geom_text(size = 2.5, hjust = -0.1, vjust = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "Age Preference Trade-off: Children vs Elderly",
       subtitle = "Countries above diagonal favor children; below favor elderly",
       x = "% Save Elderly (when present)", 
       y = "% Save Children (when present)") +
  theme_minimal()

ggsave("age_tradeoff.pdf", width = 12, height = 10)

# 4. Status preference: Doctors vs Criminals
status_preferences <- moral_dimensions %>%
  arrange(desc(n)) %>%
  head(30) %>%
  select(Country, Region, save_doctors_rate, save_criminals_rate)

ggplot(status_preferences, aes(x = save_criminals_rate, y = save_doctors_rate, 
                               color = Region, label = Country)) +
  geom_point(size = 3) +
  geom_text(size = 2.5, hjust = -0.1, vjust = 0.5) +
  labs(title = "Status Preference: Doctors vs Criminals",
       x = "% Save Criminals (when present)", 
       y = "% Save Doctors (when present)") +
  theme_minimal()

ggsave("status_preference.pdf", width = 12, height = 10)

# 5. Gender preference by region
gender_regional <- moral_dimensions %>%
  filter(!is.na(Region)) %>%
  select(Region, save_female_rate, save_male_rate) %>%
  pivot_longer(cols = c(save_female_rate, save_male_rate), 
               names_to = "gender", values_to = "save_rate") %>%
  mutate(gender = ifelse(gender == "save_female_rate", "Female", "Male"))

ggplot(gender_regional, aes(x = Region, y = save_rate, fill = gender)) +
  geom_boxplot() +
  labs(title = "Gender Preference by Region",
       x = "Region", y = "Save Rate (%)", fill = "Gender") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("gender_by_region.pdf", width = 10, height = 6)

# 6. Fitness/appearance bias: Athletes vs Large people
fitness_preferences <- moral_dimensions %>%
  arrange(desc(n)) %>%
  head(30) %>%
  select(Country, Region, save_athletes_rate, save_large_rate)

ggplot(fitness_preferences, aes(x = save_large_rate, y = save_athletes_rate, 
                                color = Region, label = Country)) +
  geom_point(size = 3) +
  geom_text(size = 2.5, hjust = -0.1, vjust = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  labs(title = "Fitness/Appearance Preference: Athletes vs Large People",
       subtitle = "Countries above diagonal favor athletes more strongly",
       x = "% Save Large People (when present)", 
       y = "% Save Athletes (when present)") +
  theme_minimal()

ggsave("fitness_preference.pdf", width = 12, height = 10)

# PART 4: Summary tables

# Country rankings for key dimensions
country_rankings <- moral_dimensions %>%
  arrange(desc(n)) %>%
  head(50) %>%
  select(Country, Region, n, save_children_rate, save_elderly_rate, 
         save_doctors_rate, save_criminals_rate, save_female_rate, save_male_rate)

write.csv(country_rankings, "country_rankings.csv", row.names = FALSE)
write.csv(regional_patterns, "regional_patterns.csv", row.names = FALSE)

# Variance analysis - which dimensions show most country variation?
dimension_variance <- moral_dimensions %>%
  summarise(across(ends_with("_rate"), ~sd(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "dimension", values_to = "std_dev") %>%
  arrange(desc(std_dev)) %>%
  mutate(dimension = gsub("save_|_rate", "", dimension))

print("Dimensions with highest cross-country variation:")
print(dimension_variance)

write.csv(dimension_variance, "dimension_variance.csv", row.names = FALSE)

cat("\n=== Scenario Analysis Complete ===\n")
cat("Files created:\n")
cat("- top20_country_patterns.pdf\n")
cat("- regional_patterns.pdf\n")
cat("- age_tradeoff.pdf\n")
cat("- status_preference.pdf\n")
cat("- gender_by_region.pdf\n")
cat("- fitness_preference.pdf\n")
cat("- country_rankings.csv\n")
cat("- regional_patterns.csv\n")
cat("- dimension_variance.csv\n")

# Select key countries for analysis

# Check top countries by response count
top_countries_list <- moral_machine_features %>%
  count(Country, sort = TRUE) %>%
  head(20)

print(top_countries_list)

# Select diverse, well-represented countries
selected_countries <- c("USA", "DEU", "CHN", "ISR", "JPN", "BRA", "IND", "SAU")

# Select countries - use top represented + diverse selection

# Check top countries by response count
top_countries_list <- moral_machine_features %>%
  count(Country, sort = TRUE) %>%
  head(20)

print(top_countries_list)

# Select: USA (largest), DEU (Western Europe), ISR (Middle East), CHN (East Asia), 
# Plus add JPN (high response count, different culture)
final_countries <- c("USA", "DEU", "ISR", "CHN", "JPN")

cat("\nFinal country selection:", paste(final_countries, collapse = ", "), "\n")

# Filter data
final_data <- moral_machine_features %>%
  filter(Country %in% final_countries)

cat("\nResponses per country:\n")
print(table(final_data$Country))

# Calculate detailed preferences

country_preferences <- final_data %>%
  group_by(Country) %>%
  summarise(
    n = n(),
    # Age
    save_elderly = sum(r_has_elderly, na.rm = TRUE) / sum(q_has_elderly, na.rm = TRUE) * 100,
    save_children = sum(r_has_child, na.rm = TRUE) / sum(q_has_child, na.rm = TRUE) * 100,
    # Gender
    save_male = sum(r_has_male, na.rm = TRUE) / sum(q_has_male, na.rm = TRUE) * 100,
    save_female = sum(r_has_female, na.rm = TRUE) / sum(q_has_female, na.rm = TRUE) * 100,
    # Fitness
    save_athletes = sum(r_has_athlete, na.rm = TRUE) / sum(q_has_athlete, na.rm = TRUE) * 100,
    save_large = sum(r_has_large, na.rm = TRUE) / sum(q_has_large, na.rm = TRUE) * 100,
    # Status
    save_doctors = sum(r_has_doctor, na.rm = TRUE) / sum(q_has_doctor, na.rm = TRUE) * 100,
    save_executives = sum(r_has_executive, na.rm = TRUE) / sum(q_has_executive, na.rm = TRUE) * 100,
    save_criminals = sum(r_has_criminal, na.rm = TRUE) / sum(q_has_criminal, na.rm = TRUE) * 100,
    # Other
    save_pets = sum(r_has_pet, na.rm = TRUE) / sum(q_has_pet, na.rm = TRUE) * 100,
    .groups = 'drop'
  )

print(country_preferences)

# Key Visualizations

# Comprehensive heatmap
preferences_matrix <- country_preferences %>%
  select(Country, starts_with("save_")) %>%
  pivot_longer(cols = starts_with("save_"), names_to = "dimension", values_to = "save_rate") %>%
  mutate(dimension = gsub("save_", "", dimension))

ggplot(preferences_matrix, aes(x = Country, y = dimension, fill = save_rate)) +
  geom_tile(color = "white", size = 1) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", 
                       midpoint = 50, na.value = "grey90") +
  geom_text(aes(label = round(save_rate, 0)), size = 4, fontface = "bold") +
  labs(title = "Cross-Country Moral Preference Heatmap",
       subtitle = paste("Countries:", paste(final_countries, collapse = ", ")),
       x = "Country", y = "Characteristic", fill = "Save Rate (%)") +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size = 11, face = "bold"),
        axis.text.y = element_text(size = 11))

ggsave("final_country_heatmap.pdf", width = 10, height = 8)

# Radar charts
preferences_long <- country_preferences %>%
  select(Country, save_elderly, save_children, save_female, save_male,
         save_athletes, save_large, save_doctors, save_criminals) %>%
  pivot_longer(cols = -Country, names_to = "dimension", values_to = "save_rate") %>%
  mutate(dimension = gsub("save_", "", dimension))

ggplot(preferences_long, aes(x = dimension, y = save_rate, group = Country, color = Country)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  facet_wrap(~Country, ncol = 3) +
  coord_polar() +
  ylim(0, 100) +
  theme_minimal() +
  theme(axis.text.x = element_text(size = 8),
        legend.position = "none",
        strip.text = element_text(size = 12, face = "bold")) +
  labs(title = "Country-Specific Moral Preference Profiles",
       y = "Save Rate (%)")

ggsave("country_radar_profiles.pdf", width = 14, height = 10)

# QA/QC: Investigate anomalous values

cat("=== INVESTIGATING CHINA CRIMINALS DATA ===\n\n")

# Check how many scenarios involve criminals in China
china_criminal_scenarios <- final_data %>%
  filter(Country == "CHN") %>%
  select(ResponseID, Question, HumanResponse, q_has_criminal, r_has_criminal)

cat("Total CHN responses:", nrow(final_data %>% filter(Country == "CHN")), "\n")
cat("Scenarios with criminals in question:", sum(china_criminal_scenarios$q_has_criminal), "\n")
cat("Scenarios where criminals were saved:", sum(china_criminal_scenarios$r_has_criminal), "\n")

# Look at the actual scenarios
china_criminal_cases <- china_criminal_scenarios %>%
  filter(q_has_criminal == TRUE)

cat("\nSample scenarios with criminals:\n")
print(head(china_criminal_cases %>% select(Question, HumanResponse), 3))

# Check the calculation
cat("\nCalculation check:\n")
cat("Numerator (criminals saved):", sum(china_criminal_scenarios$r_has_criminal, na.rm = TRUE), "\n")
cat("Denominator (criminals in scenarios):", sum(china_criminal_scenarios$q_has_criminal, na.rm = TRUE), "\n")
cat("Save rate:", sum(china_criminal_scenarios$r_has_criminal, na.rm = TRUE) / 
      sum(china_criminal_scenarios$q_has_criminal, na.rm = TRUE) * 100, "%\n")

# QAQC SHOWED INCOMPLETE DATA SO TURNING TO API MODE 

moral_machine <- read.csv("~/Desktop/moral_machine_data.csv")

cat("Loaded:", nrow(moral_machine), "rows\n")
print(table(moral_machine$Country))

# Extract features
extract_features_from_text <- function(text) {
  data.frame(
    has_elderly = grepl("elderly", text, ignore.case = TRUE),
    has_child = grepl("\\b(boy|girl|child|baby)\\b", text, ignore.case = TRUE),
    has_male = grepl("\\b(man|men|male)\\b", text, ignore.case = TRUE),
    has_female = grepl("\\b(woman|women|female)\\b", text, ignore.case = TRUE),
    has_athlete = grepl("athlete", text, ignore.case = TRUE),
    has_large = grepl("large", text, ignore.case = TRUE),
    has_doctor = grepl("doctor", text, ignore.case = TRUE),
    has_executive = grepl("executive", text, ignore.case = TRUE),
    has_criminal = grepl("criminal", text, ignore.case = TRUE),
    has_pet = grepl("\\b(dog|cat)\\b", text, ignore.case = TRUE)
  )
}

question_features <- extract_features_from_text(moral_machine$Question)
colnames(question_features) <- paste0("q_", colnames(question_features))

response_features <- extract_features_from_text(moral_machine$HumanResponse)
colnames(response_features) <- paste0("r_", colnames(response_features))

moral_machine_features <- bind_cols(moral_machine, question_features, response_features)

# Calculate preferences (minimum 15 scenarios for CHN, 20 for others)
country_preferences <- moral_machine_features %>%
  group_by(Country) %>%
  summarise(
    n = n(),
    min_threshold = ifelse(Country[1] == "CHN", 15, 20),
    save_elderly = ifelse(sum(q_has_elderly) >= min_threshold[1],
                          sum(r_has_elderly) / sum(q_has_elderly) * 100, NA),
    save_children = ifelse(sum(q_has_child) >= min_threshold[1],
                           sum(r_has_child) / sum(q_has_child) * 100, NA),
    save_male = ifelse(sum(q_has_male) >= min_threshold[1],
                       sum(r_has_male) / sum(q_has_male) * 100, NA),
    save_female = ifelse(sum(q_has_female) >= min_threshold[1],
                         sum(r_has_female) / sum(q_has_female) * 100, NA),
    save_athletes = ifelse(sum(q_has_athlete) >= min_threshold[1],
                           sum(r_has_athlete) / sum(q_has_athlete) * 100, NA),
    save_large = ifelse(sum(q_has_large) >= min_threshold[1],
                        sum(r_has_large) / sum(q_has_large) * 100, NA),
    save_doctors = ifelse(sum(q_has_doctor) >= min_threshold[1],
                          sum(r_has_doctor) / sum(q_has_doctor) * 100, NA),
    save_executives = ifelse(sum(q_has_executive) >= min_threshold[1],
                             sum(r_has_executive) / sum(q_has_executive) * 100, NA),
    save_criminals = ifelse(sum(q_has_criminal) >= min_threshold[1],
                            sum(r_has_criminal) / sum(q_has_criminal) * 100, NA),
    save_pets = ifelse(sum(q_has_pet) >= min_threshold[1],
                       sum(r_has_pet) / sum(q_has_pet) * 100, NA),
    .groups = 'drop'
  ) %>%
  select(-min_threshold)

print(country_preferences)

# Add sample size annotations
country_preferences <- country_preferences %>%
  mutate(sample_note = ifelse(n < 200, "Small sample*", ""))

cat("\n*CHN has limited data (n=172) - interpret with caution\n")
cat("KOR and JPN serve as East Asian comparisons\n")

# VIZ 

install.packages("paletteer")
library(paletteer)

scale_colour_paletteer_d("tvthemes::Alexandrite")
scale_color_paletteer_d("tvthemes::Alexandrite")
scale_fill_paletteer_d("tvthemes::Alexandrite")
paletteer_d("tvthemes::Alexandrite")


# Heatmap 
preferences_matrix <- country_preferences %>%
  select(Country, n, starts_with("save_")) %>%
  pivot_longer(cols = starts_with("save_"), names_to = "dimension", values_to = "save_rate") %>%
  mutate(
    dimension = gsub("save_", "", dimension),
    country_label = ifelse(Country == "CHN", paste0(Country, "*"), Country)
  )

ggplot(preferences_matrix, aes(x = country_label, y = dimension, fill = save_rate)) +
  geom_tile(color = "white", size = 1) +
  scale_fill_gradient2(low = "#FDC067FF", mid = "#AC9ECEFF", high = "#240E31FF", 
                       midpoint = 75, na.value = "white") +
  geom_text(aes(label = round(save_rate, 0)), size = 3.5, fontface = "bold") +
  labs(title = "Cross-Country Moral Preference Heatmap",
       subtitle = "*CHN has small sample (n=172), KOR & JPN are East Asian comparisons",
       x = "Country", y = "Characteristic", fill = "Save Rate (%)") +
  theme_minimal(base_size = 11) +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10, face = "bold"),
        axis.text.y = element_text(size = 10))

ggsave("~/Desktop/country_heatmap_with_china.pdf", width = 11, height = 8)

# 2. East Asian comparison specifically
east_asia_comparison <- country_preferences %>%
  filter(Country %in% c("CHN", "JPN", "KOR")) %>%
  select(Country, n, starts_with("save_")) %>%
  pivot_longer(cols = starts_with("save_"), names_to = "dimension", values_to = "save_rate") %>%
  mutate(dimension = gsub("save_", "", dimension))

ggplot(east_asia_comparison, aes(x = dimension, y = save_rate, fill = Country)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("CHN" = "#FDC067FF", "JPN" = "#751C6DFF", "KOR" = "#6EC5ABFF")) +
  geom_text(aes(label = round(save_rate, 0)), position = position_dodge(width = 0.9),
            vjust = -0.3, size = 2.5, color = "black") +
  coord_flip() +
  labs(title = "East Asian Countries Comparison",
       subtitle = "CHN (n=172)*, JPN (n=1,893), KOR (n=392)",
       x = "Moral Dimension", y = "Save Rate (%)") +
  theme_minimal()

ggsave("~/Desktop/east_asia_comparison.pdf", width = 10, height = 8)

# STATISTICAL SIGNIFICANCE TESTING

# Chi-square test for overall country differences
cat("1. CHI-SQUARE TEST: Are country preferences significantly different?\n\n")

# Fixed dimension names to match the actual column names
test_dimensions <- c("elderly", "child", "male", "female", "athlete", 
                     "large", "doctor", "executive", "criminal", "pet")

chi_square_results <- lapply(test_dimensions, function(dim) {
  
  # Create 2x2 tables: saved vs not saved for each country
  contingency_data <- moral_machine_features %>%
    filter(Country %in% country_preferences$Country) %>%
    group_by(Country) %>%
    summarise(
      present = sum(.data[[paste0("q_has_", dim)]], na.rm = TRUE),
      saved = sum(.data[[paste0("r_has_", dim)]], na.rm = TRUE),
      .groups = 'drop'
    ) %>%
    filter(present >= 20) %>%  # Only include if sufficient data
    mutate(not_saved = present - saved)
  
  if(nrow(contingency_data) < 2) return(NULL)
  
  # Create matrix for chi-square test
  test_matrix <- as.matrix(contingency_data[, c("saved", "not_saved")])
  rownames(test_matrix) <- contingency_data$Country
  
  # Perform test
  if(sum(test_matrix) > 0) {
    test_result <- chisq.test(test_matrix)
    
    return(data.frame(
      dimension = dim,
      chi_square = test_result$statistic,
      p_value = test_result$p.value,
      df = test_result$parameter,
      significant = ifelse(test_result$p.value < 0.001, "***",
                           ifelse(test_result$p.value < 0.01, "**",
                                  ifelse(test_result$p.value < 0.05, "*", "ns")))
    ))
  }
  return(NULL)
}) %>% bind_rows()

print(chi_square_results %>% arrange(p_value))
cat("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant\n\n")

# Pairwise comparisons with effect sizes
cat("2. PAIRWISE COUNTRY COMPARISONS\n\n")

pairwise_tests <- expand.grid(
  Country1 = country_preferences$Country,
  Country2 = country_preferences$Country,
  stringsAsFactors = FALSE
) %>%
  filter(Country1 < Country2)

pairwise_results <- lapply(1:nrow(pairwise_tests), function(i) {
  c1 <- pairwise_tests$Country1[i]
  c2 <- pairwise_tests$Country2[i]
  
  data1 <- country_preferences %>% filter(Country == c1) %>% select(starts_with("save_"))
  data2 <- country_preferences %>% filter(Country == c2) %>% select(starts_with("save_"))
  
  # Calculate absolute differences
  diffs <- abs(as.numeric(data1) - as.numeric(data2))
  
  # Calculate effect size (Cohen's d-like measure)
  pooled_sd <- sqrt((sd(as.numeric(data1), na.rm = TRUE)^2 + 
                       sd(as.numeric(data2), na.rm = TRUE)^2) / 2)
  effect_size <- mean(diffs, na.rm = TRUE) / pooled_sd
  
  data.frame(
    Country1 = c1,
    Country2 = c2,
    mean_diff = mean(diffs, na.rm = TRUE),
    max_diff = max(diffs, na.rm = TRUE),
    effect_size = effect_size,
    dimensions_differ_10plus = sum(diffs > 10, na.rm = TRUE),
    dimensions_differ_20plus = sum(diffs > 20, na.rm = TRUE)
  )
}) %>% bind_rows() %>%
  arrange(desc(mean_diff))

print(pairwise_results)

cat("\nInterpretation:\n")
cat("- mean_diff: Average difference across all moral dimensions\n")
cat("- max_diff: Largest difference on any single dimension\n")
cat("- effect_size: Standardized difference (>0.8 = large effect)\n")
cat("- Differences >10 percentage points indicate meaningful divergence\n\n")

# ANOVA for each dimension
cat("3. ANOVA: Testing if country means differ by dimension\n\n")

anova_results <- lapply(test_dimensions, function(dim) {
  
  # Prepare data for ANOVA
  anova_data <- moral_machine_features %>%
    filter(Country %in% country_preferences$Country) %>%
    filter(.data[[paste0("q_has_", dim)]] == TRUE) %>%
    mutate(response = as.numeric(.data[[paste0("r_has_", dim)]])) %>%
    select(Country, response)
  
  if(nrow(anova_data) < 50) return(NULL)
  
  # Perform ANOVA
  anova_result <- aov(response ~ Country, data = anova_data)
  summary_anova <- summary(anova_result)
  
  # Calculate eta-squared (effect size)
  ss_between <- summary_anova[[1]]["Country", "Sum Sq"]
  ss_total <- sum(summary_anova[[1]][, "Sum Sq"])
  eta_squared <- ss_between / ss_total
  
  data.frame(
    dimension = dim,
    F_statistic = summary_anova[[1]]["Country", "F value"],
    p_value = summary_anova[[1]]["Country", "Pr(>F)"],
    eta_squared = eta_squared,
    significant = ifelse(summary_anova[[1]]["Country", "Pr(>F)"] < 0.001, "***",
                         ifelse(summary_anova[[1]]["Country", "Pr(>F)"] < 0.01, "**",
                                ifelse(summary_anova[[1]]["Country", "Pr(>F)"] < 0.05, "*", "ns")))
  )
}) %>% bind_rows()

print(anova_results %>% arrange(p_value))
cat("\nη² (eta-squared): 0.01=small, 0.06=medium, 0.14=large effect\n\n")

# Export statistical results
write.csv(chi_square_results, "~/Desktop/chi_square_results.csv", row.names = FALSE)
write.csv(pairwise_results, "~/Desktop/pairwise_comparisons.csv", row.names = FALSE)
write.csv(anova_results, "~/Desktop/anova_results.csv", row.names = FALSE)

cat("\n", rep("=", 70), "\n", sep = "")
cat("STATISTICAL TESTING COMPLETE\n")
cat(rep("=", 70), "\n\n", sep = "")
cat("Ready to proceed to Machine Learning?\n")

ggplot(pairwise_results, aes(x = reorder(paste(Country1, "-", Country2), mean_diff), 
                             y = mean_diff, fill = effect_size)) +
  geom_col() +
  scale_fill_gradient2(low = "#FDC067FF", mid = "#AC9ECEFF", high = "#240E31FF",
                       midpoint = 0.3) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(title = "Pairwise Country Differences",
       subtitle = "Red line = 10% meaningful divergence threshold",
       x = "Country Pair", y = "Mean Difference Across Dimensions (%)",
       fill = "Effect Size") +
  theme_minimal()

ggsave("~/Desktop/pairwise_differences.pdf", width = 10, height = 8)

cat("\n", rep("=", 70), "\n", sep = "")
cat("STATISTICAL SUMMARY\n")
cat(rep("=", 70), "\n", sep = "")
cat("1. Overall variation is MODERATE (most p-values >0.05)\n")
cat("2. China shows distinct pattern but small sample (n=172)\n")
cat("3. Western countries (USA, DEU) are highly similar\n")
cat("4. East Asian countries (CHN, JPN, KOR) show more internal variation\n")
cat("5. No dimension shows strong effect (all η² < 0.01)\n\n")
cat("CONCLUSION: Moderate country differences suggest REGIONAL protocols\n")
cat("            may be more appropriate than fully country-specific ones\n")
cat(rep("=", 70), "\n\n", sep = "")

# MACHINE LEARNING: BUILD PERFECT AV DECISION MODELS

library(rpart)
library(rpart.plot)

# Priority-Based Decision Rules for Each Country

cat("PART 1: COUNTRY-SPECIFIC PRIORITY RANKINGS\n")
cat(rep("=", 70), "\n\n", sep = "")

# Create comprehensive priority profiles
av_decision_rules <- country_preferences %>%
  select(Country, n, starts_with("save_")) %>%
  pivot_longer(cols = starts_with("save_"), names_to = "characteristic", values_to = "priority_score") %>%
  mutate(characteristic = gsub("save_", "", characteristic)) %>%
  group_by(Country) %>%
  arrange(Country, desc(priority_score)) %>%
  mutate(
    rank = row_number(),
    priority_tier = case_when(
      rank <= 3 ~ "High Priority",
      rank <= 6 ~ "Medium Priority",
      rank <= 10 ~ "Low Priority"
    )
  ) %>%
  ungroup()

for(country in unique(av_decision_rules$Country)) {
  country_rules <- av_decision_rules %>%
    filter(Country == country) %>%
    arrange(rank)
  
  sample_size <- country_preferences %>% filter(Country == country) %>% pull(n)
  
  cat("\n", country, "AV PROTOCOL (n=", sample_size, ")\n", sep = "")
  cat(rep("-", 50), "\n", sep = "")
  
  for(i in 1:nrow(country_rules)) {
    cat(sprintf("  %d. %-12s : %5.1f%% save rate [%s]\n", 
                country_rules$rank[i],
                country_rules$characteristic[i],
                country_rules$priority_score[i],
                country_rules$priority_tier[i]))
  }
}

# PART 2: Build Decision Trees for Each Country

cat("\n\nPART 2: DECISION TREE MODELS\n")
cat(rep("=", 70), "\n\n", sep = "")

# Prepare training data - convert to binary outcomes
ml_data <- moral_machine_features %>%
  filter(Country %in% country_preferences$Country) %>%
  # Create a binary outcome: did they save the group with "higher value" characteristics?
  mutate(
    scenario_id = row_number(),
    # Simplify: Did they save children over elderly?
    saved_young = r_has_child & !r_has_elderly,
    # Did they save females over males?  
    saved_female = r_has_female & !r_has_male,
    # Did they save high-status over criminals?
    saved_high_status = (r_has_doctor | r_has_executive) & !r_has_criminal,
    # Did they save fit over large?
    saved_fit = r_has_athlete & !r_has_large
  ) %>%
  select(Country, starts_with("q_has_"), starts_with("saved_"))

# Train decision trees for each country
decision_trees <- list()

for(country in unique(ml_data$Country)) {
  cat("Training decision tree for", country, "...\n")
  
  country_data <- ml_data %>% filter(Country == country)
  
  # Train on age preference (most universal dimension)
  age_scenarios <- country_data %>%
    filter(q_has_elderly | q_has_child) %>%
    mutate(chose_youth = saved_young)
  
  if(nrow(age_scenarios) > 50) {
    tree <- rpart(
      chose_youth ~ q_has_elderly + q_has_child + q_has_male + q_has_female + 
        q_has_athlete + q_has_large + q_has_doctor + q_has_criminal,
      data = age_scenarios,
      method = "class",
      control = rpart.control(cp = 0.01, maxdepth = 4)
    )
    
    decision_trees[[country]] <- tree
    
    # Save tree plot
    pdf(paste0("~/Desktop/decision_tree_", country, ".pdf"), width = 12, height = 8)
    rpart.plot(tree, 
               main = paste(country, "AV Decision Tree: Choose Youth vs Elderly"),
               type = 3,
               extra = 104,
               under = TRUE,
               fallen.leaves = TRUE)
    dev.off()
  }
}

# PART 3: Create Weighted Scoring System for Each Country

cat("\n\nPART 3: WEIGHTED SCORING SYSTEMS\n")
cat(rep("=", 70), "\n\n", sep = "")

# Normalize scores to create weights (0-100 scale)
scoring_systems <- country_preferences %>%
  select(Country, n, starts_with("save_")) %>%
  mutate(across(starts_with("save_"), ~./100)) %>%  # Convert to 0-1 scale
  pivot_longer(cols = starts_with("save_"), names_to = "characteristic", values_to = "weight") %>%
  mutate(characteristic = gsub("save_", "", characteristic))

# Create weighted decision function
cat("WEIGHTED SCORING FORMULA FOR EACH COUNTRY:\n\n")

for(country in unique(scoring_systems$Country)) {
  country_weights <- scoring_systems %>%
    filter(Country == country) %>%
    arrange(desc(weight))
  
  cat(country, "SCORING FUNCTION:\n", sep = "")
  cat("  Score = ")
  
  formula_parts <- paste0(
    round(country_weights$weight, 2), 
    "*", 
    country_weights$characteristic
  )
  
  cat(paste(formula_parts, collapse = " + "))
  cat("\n\n")
}

# PART 4: Simulate AV Decisions for Sample Scenarios

cat("\nPART 4: SIMULATED AV DECISIONS\n")
cat(rep("=", 70), "\n\n", sep = "")

# Create test scenarios
test_scenarios <- data.frame(
  scenario = c(
    "Elderly doctor vs young criminal",
    "Female athlete vs male executive", 
    "Child with pet vs elderly person",
    "2 large people vs 1 athlete",
    "3 elderly vs 2 children"
  ),
  group_a_elderly = c(1, 0, 0, 0, 3),
  group_a_child = c(0, 0, 1, 0, 0),
  group_a_female = c(0, 1, 0, 0, 0),
  group_a_athlete = c(0, 1, 0, 0, 0),
  group_a_large = c(0, 0, 0, 2, 0),
  group_a_doctor = c(1, 0, 0, 0, 0),
  group_a_criminal = c(0, 0, 0, 0, 0),
  group_a_pet = c(0, 0, 1, 0, 0),
  group_b_elderly = c(0, 0, 1, 0, 0),
  group_b_child = c(0, 0, 0, 0, 2),
  group_b_female = c(0, 0, 0, 0, 0),
  group_b_athlete = c(0, 0, 0, 1, 0),
  group_b_large = c(0, 0, 0, 0, 0),
  group_b_doctor = c(0, 0, 0, 0, 0),
  group_b_criminal = c(1, 0, 0, 0, 0),
  group_b_pet = c(0, 0, 0, 0, 0)
)

# Calculate scores for each country
decisions <- data.frame(scenario = test_scenarios$scenario)

for(country in unique(scoring_systems$Country)) {
  country_weights <- scoring_systems %>%
    filter(Country == country) %>%
    select(characteristic, weight)
  
  # Score group A
  score_a <- sapply(1:nrow(test_scenarios), function(i) {
    sum(
      test_scenarios$group_a_elderly[i] * country_weights$weight[country_weights$characteristic == "elderly"],
      test_scenarios$group_a_child[i] * country_weights$weight[country_weights$characteristic == "children"],
      test_scenarios$group_a_female[i] * country_weights$weight[country_weights$characteristic == "female"],
      test_scenarios$group_a_athlete[i] * country_weights$weight[country_weights$characteristic == "athletes"],
      test_scenarios$group_a_large[i] * country_weights$weight[country_weights$characteristic == "large"],
      test_scenarios$group_a_doctor[i] * country_weights$weight[country_weights$characteristic == "doctors"],
      test_scenarios$group_a_criminal[i] * country_weights$weight[country_weights$characteristic == "criminals"],
      test_scenarios$group_a_pet[i] * country_weights$weight[country_weights$characteristic == "pets"]
    )
  })
  
  # Score group B
  score_b <- sapply(1:nrow(test_scenarios), function(i) {
    sum(
      test_scenarios$group_b_elderly[i] * country_weights$weight[country_weights$characteristic == "elderly"],
      test_scenarios$group_b_child[i] * country_weights$weight[country_weights$characteristic == "children"],
      test_scenarios$group_b_female[i] * country_weights$weight[country_weights$characteristic == "female"],
      test_scenarios$group_b_athlete[i] * country_weights$weight[country_weights$characteristic == "athletes"],
      test_scenarios$group_b_large[i] * country_weights$weight[country_weights$characteristic == "large"],
      test_scenarios$group_b_doctor[i] * country_weights$weight[country_weights$characteristic == "doctors"],
      test_scenarios$group_b_criminal[i] * country_weights$weight[country_weights$characteristic == "criminals"],
      test_scenarios$group_b_pet[i] * country_weights$weight[country_weights$characteristic == "pets"]
    )
  })
  
  # Decision: A or B
  decision <- ifelse(score_a > score_b, "Group A", "Group B")
  decisions[[country]] <- decision
}

print(decisions)

# PART 5: Calculate Agreement Rate Between Countries

cat("\n\nPART 5: CROSS-COUNTRY AGREEMENT ANALYSIS\n")
cat(rep("=", 70), "\n\n", sep = "")

# Calculate agreement matrix
countries <- unique(scoring_systems$Country)
agreement_matrix <- matrix(0, nrow = length(countries), ncol = length(countries))
rownames(agreement_matrix) <- countries
colnames(agreement_matrix) <- countries

for(i in 1:length(countries)) {
  for(j in 1:length(countries)) {
    c1 <- countries[i]
    c2 <- countries[j]
    agreement <- sum(decisions[[c1]] == decisions[[c2]]) / nrow(decisions) * 100
    agreement_matrix[i, j] <- agreement
  }
}

cat("AGREEMENT MATRIX (% of identical decisions):\n")
print(round(agreement_matrix, 1))

# Overall consensus
overall_agreement <- mean(agreement_matrix[lower.tri(agreement_matrix)])
cat("\nOverall cross-country agreement:", round(overall_agreement, 1), "%\n\n")

# Export everything
write.csv(av_decision_rules, "~/Desktop/av_decision_rules.csv", row.names = FALSE)
write.csv(scoring_systems, "~/Desktop/scoring_systems.csv", row.names = FALSE)
write.csv(decisions, "~/Desktop/scenario_decisions.csv", row.names = FALSE)
write.csv(agreement_matrix, "~/Desktop/agreement_matrix.csv")

cat("\n", rep("=", 70), "\n", sep = "")
cat("PERFECT AV MODELS COMPLETE\n")
cat(rep("=", 70), "\n", sep = "")
cat("\nFiles created:\n")
cat("- Decision trees (PDF) for each country\n")
cat("- av_decision_rules.csv (priority rankings)\n")
cat("- scoring_systems.csv (weighted formulas)\n")
cat("- scenario_decisions.csv (test case outcomes)\n")
cat("- agreement_matrix.csv (cross-country consensus)\n\n")

cat("FINAL ANSWER TO YOUR QUESTION:\n")
if(overall_agreement >= 90) {
  cat("Countries agree ", round(overall_agreement, 1), "% of the time\n", sep = "")
  cat("RECOMMENDATION: Universal AV protocol is feasible\n")
} else if(overall_agreement >= 75) {
  cat("Countries agree ", round(overall_agreement, 1), "% of the time\n", sep = "")
  cat("RECOMMENDATION: Universal protocol with minor regional adjustments\n")
} else {
  cat("Countries agree only ", round(overall_agreement, 1), "% of the time\n", sep = "")
  cat("RECOMMENDATION: Regional or country-specific protocols needed\n")
}

# HARD CASE STUDIES

cat("=== BUILDING 20 CHALLENGING MORAL SCENARIOS ===\n\n")

# Create 20 scenarios where weighted scores will be close
challenging_scenarios <- data.frame(
  scenario_id = 1:20,
  scenario_description = c(
    # Age trade-offs
    "3 elderly people vs 2 children",
    "1 elderly doctor vs 1 young criminal",
    "2 elderly executives vs 2 young athletes",
    
    # Gender trade-offs with other factors
    "2 males vs 3 females",
    "1 male doctor vs 1 female athlete",
    "2 male executives vs 2 female large people",
    
    # Status conflicts
    "1 doctor vs 2 executives",
    "1 executive vs 2 criminals",
    "3 homeless vs 1 doctor",
    
    # Fitness/appearance conflicts
    "3 large people vs 2 athletes",
    "1 large doctor vs 1 athletic criminal",
    "2 large elderly vs 2 athletic children",
    
    # Complex multi-factor scenarios
    "1 elderly female doctor vs 2 young male criminals",
    "1 child with pet vs 1 elderly person with pet",
    "2 male athletes vs 2 female executives",
    "3 large elderly people vs 1 athletic child",
    "1 female criminal vs 1 male homeless person",
    
    # Quantity vs quality trade-offs
    "5 criminals vs 1 doctor",
    "4 large people vs 1 child",
    "6 elderly vs 2 children with 1 pet"
  ),
  
  # Group A composition
  group_a_count = c(3, 1, 2, 2, 1, 2, 1, 1, 3, 3, 1, 2, 1, 1, 2, 3, 1, 5, 4, 6),
  group_a_elderly = c(3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 0, 6),
  group_a_child = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
  group_a_male = c(0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
  group_a_female = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
  group_a_athlete = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
  group_a_large = c(0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 1, 2, 0, 0, 0, 3, 0, 0, 4, 0),
  group_a_doctor = c(0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0),
  group_a_executive = c(0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
  group_a_criminal = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 0, 0),
  group_a_homeless = c(0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
  group_a_pet = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1),
  
  # Group B composition
  group_b_count = c(2, 1, 2, 3, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 1, 1, 1, 2),
  group_b_elderly = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
  group_b_child = c(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 2),
  group_b_male = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0),
  group_b_female = c(0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
  group_b_athlete = c(0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0),
  group_b_large = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
  group_b_doctor = c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
  group_b_executive = c(0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0),
  group_b_criminal = c(0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0),
  group_b_homeless = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
  group_b_pet = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0)
)

# CALCULATE DECISIONS FOR EACH COUNTRY

cat("Calculating AV decisions for each country...\n\n")

# Get scoring weights for each country
country_weights_list <- list()
for(country in unique(scoring_systems$Country)) {
  country_weights_list[[country]] <- scoring_systems %>%
    filter(Country == country) %>%
    select(characteristic, weight)
}

# Calculate scores and decisions
results <- data.frame(
  scenario_id = challenging_scenarios$scenario_id,
  scenario = challenging_scenarios$scenario_description
)

for(country in names(country_weights_list)) {
  weights <- country_weights_list[[country]]
  
  # Score Group A
  score_a <- sapply(1:nrow(challenging_scenarios), function(i) {
    sum(
      challenging_scenarios$group_a_elderly[i] * weights$weight[weights$characteristic == "elderly"],
      challenging_scenarios$group_a_child[i] * weights$weight[weights$characteristic == "children"],
      challenging_scenarios$group_a_male[i] * weights$weight[weights$characteristic == "male"],
      challenging_scenarios$group_a_female[i] * weights$weight[weights$characteristic == "female"],
      challenging_scenarios$group_a_athlete[i] * weights$weight[weights$characteristic == "athletes"],
      challenging_scenarios$group_a_large[i] * weights$weight[weights$characteristic == "large"],
      challenging_scenarios$group_a_doctor[i] * weights$weight[weights$characteristic == "doctors"],
      challenging_scenarios$group_a_executive[i] * weights$weight[weights$characteristic == "executives"],
      challenging_scenarios$group_a_criminal[i] * weights$weight[weights$characteristic == "criminals"],
      challenging_scenarios$group_a_pet[i] * weights$weight[weights$characteristic == "pets"],
      na.rm = TRUE
    )
  })
  
  # Score Group B
  score_b <- sapply(1:nrow(challenging_scenarios), function(i) {
    sum(
      challenging_scenarios$group_b_elderly[i] * weights$weight[weights$characteristic == "elderly"],
      challenging_scenarios$group_b_child[i] * weights$weight[weights$characteristic == "children"],
      challenging_scenarios$group_b_male[i] * weights$weight[weights$characteristic == "male"],
      challenging_scenarios$group_b_female[i] * weights$weight[weights$characteristic == "female"],
      challenging_scenarios$group_b_athlete[i] * weights$weight[weights$characteristic == "athletes"],
      challenging_scenarios$group_b_large[i] * weights$weight[weights$characteristic == "large"],
      challenging_scenarios$group_b_doctor[i] * weights$weight[weights$characteristic == "doctors"],
      challenging_scenarios$group_b_executive[i] * weights$weight[weights$characteristic == "executives"],
      challenging_scenarios$group_b_criminal[i] * weights$weight[weights$characteristic == "criminals"],
      challenging_scenarios$group_b_pet[i] * weights$weight[weights$characteristic == "pets"],
      na.rm = TRUE
    )
  })
  
  # Decision and margin
  results[[paste0(country, "_decision")]] <- ifelse(score_a > score_b, "A", 
                                                    ifelse(score_b > score_a, "B", "TIE"))
  results[[paste0(country, "_margin")]] <- abs(score_a - score_b)
}

# ANALYZE DISAGREEMENTS

cat("=== DECISION ANALYSIS ===\n\n")

# Create decision matrix (just the decisions)
decision_cols <- grep("_decision$", names(results), value = TRUE)
decision_matrix <- results[, c("scenario_id", "scenario", decision_cols)]

# Simplify column names
names(decision_matrix) <- gsub("_decision", "", names(decision_matrix))

print(decision_matrix)

# Count disagreements per scenario
results$disagreement_count <- apply(results[, decision_cols], 1, function(x) {
  length(unique(x[!is.na(x)])) - 1
})

# Identify controversial scenarios
controversial <- results %>%
  filter(disagreement_count > 0) %>%
  select(scenario_id, scenario, disagreement_count, ends_with("_decision"))

cat("\n=== CONTROVERSIAL SCENARIOS (Countries Disagree) ===\n\n")
if(nrow(controversial) > 0) {
  print(controversial)
} else {
  cat("No disagreements found!\n")
}

# Calculate pairwise disagreement rates
countries <- gsub("_decision", "", decision_cols)
disagreement_matrix <- matrix(0, nrow = length(countries), ncol = length(countries))
rownames(disagreement_matrix) <- countries
colnames(disagreement_matrix) <- countries

for(i in 1:length(countries)) {
  for(j in 1:length(countries)) {
    c1 <- countries[i]
    c2 <- countries[j]
    disagreements <- sum(results[[paste0(c1, "_decision")]] != results[[paste0(c2, "_decision")]], na.rm = TRUE)
    disagreement_matrix[i, j] <- disagreements
  }
}

cat("\n=== PAIRWISE DISAGREEMENT MATRIX (# of scenarios where countries differ) ===\n\n")
print(disagreement_matrix)

# Overall agreement rate
total_comparisons <- nrow(results) * length(countries) * (length(countries) - 1) / 2
total_disagreements <- sum(disagreement_matrix[lower.tri(disagreement_matrix)])
agreement_rate <- (1 - total_disagreements / (nrow(results) * choose(length(countries), 2))) * 100

cat("\n", rep("=", 70), "\n", sep = "")
cat("FINAL RESULTS: CHALLENGING SCENARIOS\n")
cat(rep("=", 70), "\n", sep = "")
cat("Total scenarios tested:", nrow(results), "\n")
cat("Scenarios with ANY disagreement:", sum(results$disagreement_count > 0), "\n")
cat("Overall agreement rate:", round(agreement_rate, 1), "%\n\n")

if(agreement_rate >= 90) {
  cat("CONCLUSION: Countries agree ", round(agreement_rate, 1), "% of the time\n", sep = "")
  cat("Even in challenging moral dilemmas, values align closely.\n")
  cat("RECOMMENDATION: Universal AV protocol is feasible\n\n")
} else if(agreement_rate >= 75) {
  cat("CONCLUSION: Countries agree ", round(agreement_rate, 1), "% of the time\n", sep = "")
  cat("Most decisions align, but some key differences exist.\n")
  cat("RECOMMENDATION: Universal protocol with regional adjustments\n\n")
} else {
  cat("CONCLUSION: Countries agree only ", round(agreement_rate, 1), "% of the time\n", sep = "")
  cat("Significant moral disagreements exist.\n")
  cat("RECOMMENDATION: Country-specific or regional protocols needed\n\n")
}

# CRITIQUE VIZ 

test_dimensions <- c("elderly", "child", "male", "female", 
                     "athlete", "large", "doctor", "executive", 
                     "criminal", "pet")

gg_eta <- ggplot(anova_plot_data,
                 aes(x = dimension, y = eta_squared)) +
  geom_col(fill = "#AC9ECE") +
  geom_hline(yintercept = 0.01, linetype = "dashed", color = "red") +
  geom_text(aes(label = round(eta_squared, 3)),
            vjust = -0.3, size = 3) +
  coord_flip() +
  labs(
    title    = "Effect Sizes by Moral Dimension (ANOVA)",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x        = "Moral Dimension",
    y        = "Eta-squared (η²: <0.01 = negligible effect)"
  ) +
  theme_minimal()

gg_eta

gg_chi <- ggplot(chi_plot_data,
                 aes(x = dimension, y = neg_log10_p, fill = significant)) +
  geom_col() +
  geom_hline(yintercept = -log10(0.05),
             linetype = "dashed", color = "red") +
  scale_fill_manual(values = c("ns"="#AC9ECE","*"="#FDC067","**"="#751C6D","***"="#240E31")) +
  coord_flip() +
  labs(
    title    = "Chi-square Tests across Countries",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x        = "Moral Dimension",
    y        = "-log10(p-value)",
    fill     = "Significance"
  ) +
  theme_minimal()

gg_chi

gg_pairwise <- ggplot(pairwise_results,
                      aes(x = reorder(paste(Country1, "-", Country2), mean_diff),
                          y = mean_diff, fill = effect_size)) +
  geom_col() +
  scale_fill_gradient2(
    low="#FDC067", mid="#AC9ECE", high="#240E31", midpoint=0.3
  ) +
  geom_hline(yintercept = 10, linetype = "dashed", color = "red") +
  coord_flip() +
  labs(
    title    = "Pairwise Country Differences in Save-Rate Preferences",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x        = "Country Pair",
    y        = "Mean Difference (%)",
    fill     = "Effect Size"
  ) +
  theme_minimal()

gg_pairwise

gg_av <- ggplot(av_plot_data,
                aes(x = characteristic,
                    y = priority_score,
                    fill = priority_tier)) +
  geom_col() +
  geom_text(aes(label = paste0(round(priority_score, 1), "%")),
            hjust=-0.1, size=2.7) +
  coord_flip() +
  facet_wrap(~ Country, ncol=3) +
  scale_fill_manual(values = c(
    "High Priority"="#751C6D",
    "Medium Priority"="#6EC5AB",
    "Low Priority"="#FDC067"
  )) +
  labs(
    title    = "AV Protocol Priority Rankings by Country",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x        = "Characteristic",
    y        = "Save Rate (%)",
    fill     = "Priority Tier"
  ) +
  theme_minimal() +
  theme(legend.position="bottom")

gg_av

bump_data <- av_decision_rules %>%
  select(Country, characteristic, rank) %>%
  mutate(
    characteristic = factor(characteristic),
    Country = factor(Country, levels = c("CHN", "KOR", "JPN", "ISR", "DEU", "USA"))
  )

gg_bump <- ggplot(bump_data,
                  aes(x = Country,
                      y = rank,
                      group = characteristic,
                      color = characteristic)) +
  geom_line(size = 1.1, alpha = 0.9) +
  geom_point(size = 2) +
  scale_y_reverse(breaks = 1:10) +
  labs(
    title = "Cross-Country Ranking of AV Priorities",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x = "Country",
    y = "Rank (1 = highest priority)",
    color = "Characteristic"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

gg_bump

heat_data <- av_decision_rules %>%
  mutate(
    characteristic = factor(
      characteristic,
      levels = rev(c("doctors", "executives", "children",
                     "female", "male", "athletes",
                     "criminals", "large", "elderly", "pets"))
    ),
    Country = factor(Country, levels = c("CHN","KOR","JPN","ISR","DEU","USA"))
  )

gg_heat <- ggplot(heat_data,
                  aes(x = Country, y = characteristic, fill = rank)) +
  geom_tile(color = "black") +
  geom_text(aes(label = rank), color = "white", size = 3) +
  scale_fill_gradient(low = "#FDC067", high = "#751C6D",
                      guide = guide_colorbar(title = "Rank\n(1 = highest)")) +
  scale_y_discrete(expand = c(0,0)) +
  scale_x_discrete(expand = c(0,0)) +
  labs(
    title    = "Implied AV Priority Ranks by Country",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x        = "Country",
    y        = "Characteristic"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid = element_blank()
  )

gg_heat

bar_data <- av_decision_rules %>%
  mutate(
    characteristic = factor(
      characteristic,
      levels = c("doctors", "executives", "children",
                 "female", "male", "athletes",
                 "criminals", "large", "elderly", "pets")
    ),
    Country = factor(Country, levels = c("CHN","KOR","JPN","ISR","DEU","USA"))
  )

gg_bar <- ggplot(bar_data,
                 aes(x = characteristic, y = priority_score,
                     fill = priority_tier)) +
  geom_col() +
  geom_text(aes(label = paste0(round(priority_score, 1), "%")),
            vjust = -0.2, size = 2.5) +
  facet_wrap(~ Country, ncol = 3) +
  scale_fill_manual(values = c(
    "High Priority"   = "#751C6D",
    "Medium Priority" = "#6EC5AB",
    "Low Priority"    = "#FDC067"
  )) +
  labs(
    title    = "Save Rates by Characteristic and Country",
    subtitle = "*CHN has small sample (n=172); KOR & JPN are East Asian comparisons",
    x        = "Characteristic",
    y        = "Save Rate (%)",
    fill     = "Priority Tier"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

gg_bar
