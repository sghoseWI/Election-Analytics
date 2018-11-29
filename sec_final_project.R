#Saptarshi Ghose + Natasha Mathur
#Science of Elections and Campaigns
#Final Project

rm(list=ls())   

#Set working directory
setwd("~/Downloads/Texas-Blue-master 5")

#Install libraries
install.packages('plyr')
install.packages('lfe')
install.packages('stargazer')

#Load libraries
library(tidyverse)
library(ggplot2)
library(haven)
library(plyr)
library(stargazer)

options(scipen=999)

#Import dta file
data <- read_csv(file = "final_data_for_model.csv")

#Attach the data
attach(data)

#models
model1 <- lm(vote_share ~ pro_gun_control + candidate_hispanic  + factor(year) + factor(office) + factor(county) + factor(district))
summary(model1)

model2 <- lm(vote_share ~ pro_gun_control + candidate_male + candidate_hispanic +   factor(year) + factor(office) + factor(county) + factor(district))
summary(model2)

stargazer(model1, model2, type = 'text', style = 'jpam')


model3 <- lm(vote_share ~ pro_gun_control + candidate_male + candidate_hispanic + factor(county) + factor(district) + factor(year))
summary(model3)

# model4 <- lm(vote_share ~ pro_gun_control + pro_choice + candidate_female + candidate_hispanic + urban + percent_hispanic + factor(county_num_x) + factor(district_num_x))
# summary(model2)
# 
# model5- lm(vote_share ~ pro_gun_control + candidate_female + candidate_hispanic + urban + percent_hispanic + factor(county_num_x) + factor(district_num_x))
# summary(model5)
# 
# model6 <- lm(vote_share ~ pro_gun_control + pro_choice + candidate_female + candidate_hispanic + urban + percent_hispanic + factor(county_num_x) + factor(district_num_x))
# summary(model6)



