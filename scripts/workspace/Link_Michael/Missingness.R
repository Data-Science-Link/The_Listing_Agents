library(dplyr)
library(ggplot2)
library(tidyr)
library(VIM)
library(Hmisc)

# Importing the two kaggle datasets
train = read.csv('/Users/michaellink/Desktop/__NYCDSA/_Projects/Machine_Learning/data/kaggle/train.csv')
test = read.csv('/Users/michaellink/Desktop/__NYCDSA/_Projects/Machine_Learning/data/kaggle/test.csv')

# Taking first glance at missingness
Missingness_Summary_Graphic = VIM::aggr(train, cex.axis = .9, oma = c(10,5,5,3))


# Initiating blank columns
ColName=character(0)
NumNA=integer(0)
PercentNA=numeric(0)
Classy=character(0)
First_val=character(0)

# Creating for loop to calculate NumNA's, percent NA's, the class, and an example value to determine true class
for (val in 1:length(colnames(train))) {
  ColName[val] = colnames(train)[val]
  NumNA[val] = sum(is.na(train[val]))
  PercentNA[val] = round(((sum(is.na(train[val])) / nrow(train[val])) * 100), 1)
  Classy[val] = lapply(train, class)[[val]]
  First_val[val] = as.character(train[colnames(train)[val]][50,1])
}

# Aggregating above columns to create a table of missingness
Missingness_percent_df = data.frame(ColName, NumNA, PercentNA, Classy, First_val) %>% 
  filter(., PercentNA > 0) %>% 
  arrange(., desc(NumNA))
print(Missingness_percent_df)

#Write to CSV
write.csv(Missingness_percent_df,
          '/Users/michaellink/Desktop/__NYCDSA/_Projects/Machine_Learning/data/missingness/missingness_summary.csv'
          )

# Only two columns of numerical missingness. GarageYrBlt and LotFrontage. GarageYrBuilt is NA for a reason. Investigating LotFrontage.

# Imputing randomly and using mean
imputed_LotFrontage = impute(train$LotFrontage, 'random')
mean_before_imputation = mean(train$LotFrontage, na.rm = TRUE)

# Creating new columns to incorporate imputation and then pivot longer in preparation of GGPlot Tidy data format
train_mean_value_imputation = 
  train %>% 
  mutate(., LotFrontage_Mean = ifelse(is.na(LotFrontage), mean_before_imputation, LotFrontage)) %>% 
  mutate(., LotFrontage_Random = as.numeric(imputed_LotFrontage)) %>% 
  pivot_longer(c(LotFrontage, LotFrontage_Mean, LotFrontage_Random), names_to = 'type', values_to = 'values')

g = ggplot(data = train_mean_value_imputation) + 
  geom_density(aes(x = values, colour = type)) +
  labs(title = 'LotFrontage Imputation (Linear feet of street connected to property)', x = 'Value', y = 'Density') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5))
g




# colnames(train)[5]
# class(train['GarageType'][50,1])
# train[colnames(train)[5]][1,1]
# 
# train['GarageType'][50,1]
# levels(train['GarageType'])
# lapply(train, levels)[[1]]



# 
# # Missingness_percent_no_NA_df = Missingness_percent_df %>% 
# #   filter(., !is.na(First_val))
# # print(Missingness_percent_no_NA_df)
# 
# 
# 
# ##########################################
# 
# pivot_longer(data = Missingness_percent_no_NA_df, 
#              cols = colnames(Missingness_percent_no_NA_df), 
#              names_to = 'Variable', values_to = 'Value', 
#              values_ptypes = list(Value = 'character')
# )
# 
# pivot_longer(data = train_select, 
#              cols = colnames(train_select), 
#              names_to = 'Variable', values_to = 'Value', 
#              values_ptypes = list(Value = 'character')
#              )


















