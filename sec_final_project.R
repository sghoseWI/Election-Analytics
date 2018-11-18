#Saptarshi Ghose + Natasha Mathur
#Science of Elections and Campaigns
#Final Project

rm(list=ls())   

#Set working directory
setwd("~/Downloads")

#Install libraries
install.packages('plyr')
install.packages('lfe')

#Load libraries
library(tidyverse)
library(ggplot2)
library(haven)
library(plyr)

#Import dta file
data <- read_csv(file = "data_for_model.csv")

#Attach the data
attach(data)

#models
model1 <- lm(two_party_vtshare ~ pro_gun_control + pro_choice + candidate_male + candidate_hispanic + urban + percent_hispanic + factor(county_num_x) + factor(district_num_x))
summary(model1)

model2 <- lm(two_party_vtshare ~ pro_gun_control + pro_choice + candidate_female + candidate_hispanic + urban + percent_hispanic + factor(county_num_x) + factor(district_num_x))
summary(model2)


